import argparse
import os
import subprocess
from pathlib import Path
import numpy as np


TARGET_EXTENSION = [".mp4"]
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
USING_CUDA = True
VIDEO_ENCODER = "hevc_nvenc"

THRESHOLD_PSNR = 40.0
THRESHOLD_SSIM = 0.93
COMPRESS_RATIO = [0.6, 0.7, 0.8, 0.9]

class ContentTranscoding:
    def __init__(self, args):
        self.args = args
        self.tmp_file = "_tmp.mp4"
        
    def gethering_target_files(self):
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

        cmd = f"{FFMPEG} -y {using_hwaccel} -i {target_file} -b:v {video_bitrate} -c:v {video_encoder} -c:a copy {self.tmp_file}"
        try:
            subprocess.call(cmd, shell=True)
        except subprocess.CalledProcessError:
            print(f"[Error] Failed to transcode the video file = {target_file} and bitrate = {video_bitrate}")

    def __measuring(self, anchor_file, target_file):
        cmd = f'{FFMPEG} -i {anchor_file} -i {target_file} -filter_complex \"[1:v:0]split=2[ref1][ref2];[0:v:0][ref1]psnr=f=psnr_report.txt[v_pass];[v_pass][ref2]ssim=f=ssim_report.txt\" -f null -'
        try:
            subprocess.call(cmd, shell=True)
        except subprocess.CalledProcessError:
            print(f"[Error] Failed to measure the video file = {target_file}")

    def __parsing_psnr_ssim(self):
        try:
            parsed_data = []
            with open('psnr_report.txt', 'r', encoding='utf-8') as file:
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

            with open('ssim_report.txt', 'r', encoding='utf-8') as file:
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
                            'All': float(line_dict['ssim_all']),
                            'Y': float(line_dict['ssim_y'])
                        })

            psnr_avg_list = [data['psnr_avg'] for data in parsed_data]
            psnr_y_list = [data['psnr_y'] for data in parsed_data]
            avg_psnr_avg = np.mean(psnr_avg_list)
            avg_psnr_y = np.mean(psnr_y_list)
            
            ssim_all_list = [data['ssim_all'] for data in parsed_data]
            ssim_y_list = [data['ssim_y'] for data in parsed_data]
            avg_ssim_all = np.mean(ssim_all_list)
            avg_ssim_y = np.mean(ssim_y_list)
        except FileNotFoundError:
            print("Can't find psnr_report.txt. please check path.")
            avg_psnr_avg, avg_psnr_y = 0, 0
        
        return avg_psnr_avg, avg_psnr_y, avg_ssim_all, avg_ssim_y
    
    def run_transcoding(self, target_files):
        for cur_file in target_files:
            cmd_get_bitrate = f"{FFPROBE} -v error -show_entries format=bit_rate -of default=noprint_wrappers=1:nokey=1 {cur_file}"
            try:
                orig_video_bitrate = subprocess.check_output(cmd_get_bitrate, shell=True).decode("utf-8").strip()
                for ratio in COMPRESS_RATIO:
                    video_bitrate = int(orig_video_bitrate) * ratio
                    self.__transcoding(cur_file, video_bitrate)
                    self.__measuring(cur_file, self.tmp_file)
                    avg_psnr_avg, avg_psnr_y, avg_ssim_all, avg_ssim_y = self.__parsing_psnr_ssim()
                    if avg_psnr_avg > THRESHOLD_PSNR and avg_psnr_y > THRESHOLD_PSNR and avg_ssim_all > THRESHOLD_SSIM and avg_ssim_y > THRESHOLD_SSIM:
                        break
                    
            except subprocess.CalledProcessError:
                print("[Error] Failed to get bitrate of the video file = {}".format(cur_file))
            
    def run(self):
        target_files = self.gethering_target_files()
        self.run_transcoding(target_files)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to the video file")
    
    args = parser.parse_args()
    
    content_transcoding = ContentTranscoding(args)
    content_transcoding.run()
