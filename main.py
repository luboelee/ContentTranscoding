import ContentTranscoding as ct


import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to the video file")
    parser.add_argument("--psnr", default=ct.THRESHOLD_PSNR, type=float, help="Threshold of PSNR")
    parser.add_argument("--ssim", default=ct.THRESHOLD_SSIM, type=float, help="Threshold of SSIM")
    parser.add_argument("-t", "--threshold", action="store_true", help="Show threshold of PSNR and SSIM")
    args = parser.parse_args()
    ct.THRESHOLD_PSNR = args.psnr
    ct.THRESHOLD_SSIM = args.ssim

    if args.threshold:
        print(f"Threshold of PSNR: {ct.THRESHOLD_PSNR}, Threshold of SSIM: {ct.THRESHOLD_SSIM}")
        exit(0)

    if args.path == None:
        parser.print_help()
        exit(0)


    content_transcoding = ct.ContentTranscoding(args)
    content_transcoding.run()
