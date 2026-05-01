import argparse
import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import shutil


TARGET_EXTENSION = [".mp4"]
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
USING_CUDA = True

THRESHOLD_PSNR = 40.0
THRESHOLD_SSIM = 0.93
COMPRESS_RATIO = [0.6, 0.7, 0.8, 0.9]

class ContentTranscoding:
    def __init__(self, args):
        self.args = args
        self.temp_path = None
        self.done_path = None
        self.orig_target_file_size = []

    def __prepare(self, target_path):
        self.temp_path = Path(target_path) / "temporary"
        if os.path.isdir(self.temp_path) == False:
            os.mkdir(self.temp_path)

        self.done_path = Path(target_path) / "done"
        if os.path.isdir(self.done_path) == False:
            os.mkdir(self.done_path)

    def __gethering_target_files(self):
        target_path = Path(self.args.path)
        target_files = []
        for ext in TARGET_EXTENSION:
            file = target_path.glob(f"*{ext}")
            for x in file:
                target_files.append(x)
        return target_files

    def __transcoding(self, target_file, video_bitrate):
        if USING_CUDA == True:
            video_encoder = "hevc_nvenc"
            using_hwaccel = "-hwaccel cuda"
        else:
            video_encoder = "libx265"
            using_hwaccel = ""

        transcoded_file = self.temp_path / target_file.name
        cmd = f"{FFMPEG} -y -loglevel error {using_hwaccel} -i {target_file} -b:v {video_bitrate} -c:v {video_encoder} -c:a copy {transcoded_file}"
        try:
            subprocess.call(cmd, shell=True)
        except subprocess.CalledProcessError:
            print(f"[Error] Failed to transcode the video file = {target_file} and bitrate = {video_bitrate}")
            transcoded_file = None
        return transcoded_file

    def __measuring(self, anchor_file, target_file):
        if target_file == None:
            return

        psnr_report = f"{target_file.name}_psnr.txt"
        ssim_report = f"{target_file.name}_ssim.txt"
        cmd = f'{FFMPEG} -loglevel error -i {anchor_file} -i {target_file} -filter_complex \"[1:v:0]split=2[ref1][ref2];[0:v:0][ref1]psnr=f={str(psnr_report)}[v_pass];[v_pass][ref2]ssim=f={str(ssim_report)}\" -f null -'
        try:
            subprocess.call(cmd, shell=True)
            moved_psnr_path_file = self.temp_path / psnr_report
            moved_ssim_path_file = self.temp_path / ssim_report
            shutil.move(psnr_report, moved_psnr_path_file)
            shutil.move(ssim_report, moved_ssim_path_file)
        except subprocess.CalledProcessError:
            print(f"[Error] Failed to measure the video file = {target_file}")
            moved_psnr_path_file = None
            moved_ssim_path_file = None
        return moved_psnr_path_file, moved_ssim_path_file

    def __parsing_psnr_ssim(self, psnr_report, ssim_report):
        try:
            parsed_data = []
            with open(psnr_report, 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split()

                    line_dict = {}
                    for part in parts:
                        if ':' in part:
                            key, value = part.split(':', 1)
                            line_dict[key] = value

                    if 'psnr_avg' in line_dict and 'psnr_y' in line_dict:
                        parsed_data.append({
                            'frame': int(line_dict.get('n', 0)),
                            'psnr_avg': float(line_dict['psnr_avg']),
                            'psnr_y': float(line_dict['psnr_y'])
                        })

            psnr_avg_list = [data['psnr_avg'] for data in parsed_data]
            psnr_y_list = [data['psnr_y'] for data in parsed_data]
            avg_psnr_avg = np.mean(psnr_avg_list)
            avg_psnr_y = np.mean(psnr_y_list)

            parsed_data.clear()
            with open(ssim_report, 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split()
                    line_dict = {}

                    for part in parts:
                        if ':' in part:
                            key, value = part.split(':', 1)
                            line_dict[key] = value

                    if 'All' in line_dict and 'Y' in line_dict:
                        parsed_data.append({
                            'frame': int(line_dict.get('n', 0)),
                            'ssim_all': float(line_dict['All']),
                            'ssim_y': float(line_dict['Y'])
                        })

            ssim_all_list = [data['ssim_all'] for data in parsed_data]
            ssim_y_list = [data['ssim_y'] for data in parsed_data]
            avg_ssim_all = np.mean(ssim_all_list)
            avg_ssim_y = np.mean(ssim_y_list)
        except FileNotFoundError:
            print("Can't find psnr_report.txt. please check path.")
            avg_psnr_avg, avg_psnr_y = 0, 0

        return np.round(avg_psnr_avg, 3), np.round(avg_psnr_y, 3), np.round(avg_ssim_all, 6), np.round(avg_ssim_y, 6)

    def list_up_already_measured_files(self):
        empty_files = [f for f in self.temp_path.iterdir() if f.is_file() and f.stat().st_size == 0]
        for rm_file in empty_files:
            rm_file.unlink(missing_ok=True)

        txt_files = self.temp_path.glob("*.txt")
        mp4_files = self.temp_path.glob("*.mp4")

        measured_txt_files = []
        for x in txt_files:
            pos = x.name.find("_psnr.txt")
            if pos == -1:
                pos = x.name.find("_ssim.txt")
            if pos > 0 and x.name[:pos] not in measured_txt_files:
                measured_txt_files.append(x.name[:pos])

        already_measured_files = []
        for x in mp4_files:
            if x.name in measured_txt_files and x.stat().st_size > 0:
                already_measured_files.append(x.name)

        return already_measured_files

    def __remove_files(self, target_files):
        for x in target_files:
            x.unlink(missing_ok=True)

    def __get_original_transcoded_file_size(self, transcoded_file):
        for x in self.orig_target_file_size:
            if x[1] == transcoded_file:
                return x[0].stat().st_size, x[1].stat().st_size
        return 1, 1

    def __gethering_measured_data(self):

        transcoded_mp4_files = self.temp_path.glob("*.mp4")
        all_measured_files = []
        results = []
        for transcoded_file in transcoded_mp4_files:
            psnr_file = transcoded_file.with_name(f"{transcoded_file.name}_psnr.txt")
            ssim_file = transcoded_file.with_name(f"{transcoded_file.name}_ssim.txt")
            all_measured_files.append(psnr_file)
            all_measured_files.append(ssim_file)
            avg_psnr_avg, avg_psnr_y, avg_ssim_all, avg_ssim_y = self.__parsing_psnr_ssim(psnr_file, ssim_file)
            orig_file_size, trans_file_size = self.__get_original_transcoded_file_size(transcoded_file)
            new_data = {
                'file_name': transcoded_file.name,
                'psnr_avg': avg_psnr_avg,
                'psnr_y': avg_psnr_y,
                'ssim_all': avg_ssim_all,
                'ssim_y': avg_ssim_y,
                "orig_file_size": orig_file_size,
                "trans_file_size": trans_file_size,
                "ratio": trans_file_size / orig_file_size
            }
            results.append(new_data)

        df = pd.DataFrame(results)
        dst_path_file = self.done_path / "measured_data.csv"
        count = 0
        while True:
            if os.path.isfile(dst_path_file):
                dst_path_file = self.done_path / f"measured_data_{count}.csv"
                count += 1
            else:
                break

        try:
            df.to_csv(dst_path_file, index=False)
            print(f"[S] Saved to {dst_path_file}")
            self.__remove_files(all_measured_files)
        except Exception as e:
            print(f"[E] Failed to save to {dst_path_file}")
            return False
        return True

    def __move_transcoded_files(self):
        mp4_files = self.temp_path.glob("*.mp4")
        fail_count = 0
        for file in mp4_files:
            try:
                shutil.move(str(file), self.done_path)
            except Exception as e:
                print(f"[E] Failed to move {file} to {self.done_path}")
                fail_count += 1

        if fail_count == 0:
            print(f"[S] Move all transcoded files to {self.done_path}")
        else:
            print(f"[E] Failed to move {fail_count} files to {self.done_path}")

        if not any(self.temp_path.iterdir()):
            self.temp_path.rmdir()

    def __run_transcoding(self, target_files):
        already_measured_files = self.list_up_already_measured_files()

        count = 0
        for cur_file in target_files:
            count += 1
            if cur_file.name in already_measured_files:
                print(f"[Skip] {cur_file.name}. because already measured")
                test_file_size = [cur_file, self.temp_path / cur_file.name]
                self.orig_target_file_size.append(test_file_size)
                continue
            try:
                cmd_get_bitrate = f"{FFPROBE} -v error -show_entries format=bit_rate -of default=noprint_wrappers=1:nokey=1 {cur_file}"
                orig_video_bitrate = subprocess.check_output(cmd_get_bitrate, shell=True).decode("utf-8").strip()
                transcoding_done = False
                for ratio in COMPRESS_RATIO:
                    video_bitrate = int(orig_video_bitrate) * ratio
                    print(f"[{count}/{len(target_files)}] Transcoding the video file = {cur_file} and original bitrate = {orig_video_bitrate}, target bitrate = {video_bitrate}")
                    transcoded_file = self.__transcoding(cur_file, video_bitrate)
                    if transcoded_file == None:
                        continue
                    psnr_report, ssim_report = self.__measuring(cur_file, transcoded_file)
                    avg_psnr_avg, avg_psnr_y, avg_ssim_all, avg_ssim_y = self.__parsing_psnr_ssim(psnr_report, ssim_report)
                    if avg_psnr_avg > THRESHOLD_PSNR and avg_psnr_y > THRESHOLD_PSNR and avg_ssim_all > THRESHOLD_SSIM and avg_ssim_y > THRESHOLD_SSIM:
                        print(f"[✔] Done transcoding. Avg PSNR: {avg_psnr_avg}, Avg PSNR Y: {avg_psnr_y}, Avg SSIM All: {avg_ssim_all}, Avg SSIM Y: {avg_ssim_y}")
                        transcoding_done = True
                        test_file_size = [cur_file, transcoded_file]
                        self.orig_target_file_size.append(test_file_size)
                        break
                    else:
                        print(f"[✘] Substandard video quality: Avg PSNR: {avg_psnr_avg}/{THRESHOLD_PSNR}, Avg PSNR Y: {avg_psnr_y}/{THRESHOLD_PSNR}, Avg SSIM All: {avg_ssim_all}/{THRESHOLD_SSIM}, Avg SSIM Y: {avg_ssim_y}/{THRESHOLD_SSIM}")
                if transcoding_done == False:
                    transcoded_file.unlink(missing_ok=True)
                    psnr_report.unlink(missing_ok=True)
                    ssim_report.unlink(missing_ok=True)

            except subprocess.CalledProcessError:
                print("[Error] Failed to get bitrate of the video file = {}".format(cur_file))

    def run(self):

        self.__prepare(self.args.path)
        target_files = self.__gethering_target_files()
        self.__run_transcoding(target_files)
        if self.__gethering_measured_data() == True:
            self.__move_transcoded_files()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to the video file")
    args = parser.parse_args()

    content_transcoding = ContentTranscoding(args)
    content_transcoding.run()
