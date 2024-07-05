import argparse
import configparser
import os
from datetime import datetime
import sys
import colorama
from tqdm import tqdm
colorama.init(strip=not sys.stdout.isatty())
from termcolor import cprint
from pyfiglet import figlet_format

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert an audio file to WAV format')
    parser.add_argument('file_path', help='Path to the file or directory to convert')
    parser.add_argument('-output_path', help='Path to the output directory', default=os.getcwd() + os.sep + 'output')
    parser.add_argument('-project_name', help='project_name', type=str, default='none')
    parser.add_argument('-whisper_model', help='Which whisper model to use, HF repo address', default='openai/whisper-large-v3')
    parser.add_argument('-device', help='Which device to use, cpu or cuda', default='cuda')
    parser.add_argument('-mode', help='Automatic diarization and segmentation', default='segment and transcribe', type=str, choices=['auto', 'segment and transcribe', 'diarize', 'transcribe'])
    parser.add_argument('-diaization_model', help='Which diarization model to use, HF repo address', default='pyannote/speaker-diarization')
    parser.add_argument('-segmentation_model', help='Which segmentation model to use, HF repo address', default='pyannote/segmentation')
    parser.add_argument('-seg_onset', help='onset activation threshold, influences the segment detection', default=0.6, type=float)
    parser.add_argument('-seg_offset', help='offset activation threshold, influences the segment detection', default=0.9, type=float)
    parser.add_argument('-seg_min_duration', help='minimum duration of a segment, remove speech regions shorter than that many seconds.', default=2.0, type=float)
    parser.add_argument('-seg_min_duration_off', help='fill non-speech regions shorter than that many seconds.', default=0.0, type=float)
    parser.add_argument('-hf_token', help='Huggingface token', default='')
    parser.add_argument('-valid_ratio', help='Ratio of validation data', default=0.2, type=float)
    parser.add_argument('-ignore-cofnig', help='Ignore the config, specifiy your own setting sin CLI', action='store_true')
    args = parser.parse_args()

    project_timestamp = generate_timestamp()

    if not args.ignore_cofnig:
        config = configparser.ConfigParser()
        config.read('config.ini')
        cfg = config['default']
        args.project_name = cfg.get('project_name', args.project_name)
        args.device = cfg.get('device', args.device)
        args.mode = cfg.get('mode', args.mode)
        args.diaization_model = cfg.get('diaization_model', args.diaization_model)
        args.segmentation_model = cfg.get('segmentation_model', args.segmentation_model)
        args.hf_token = cfg.get('hf_token', args.hf_token)
        args.valid_ratio = cfg.get('valid_ratio', args.valid_ratio)
        args.whisper_model = cfg.get('whisper_model', args.whisper_model)

    # Create the output structure
    output_path = create_output_structure(args.output_path, args.project_name, project_timestamp)

    cprint(figlet_format('PROCESIZNG', font='starwars'), 'yellow', 'on_blue', attrs=['bold'])

    # Convert to WAV if necessary
    original_file_path = args.file_path
    if not original_file_path.endswith('.wav'):
        args.file_path = convert_to_wav(args.file_path)
    else:
        audio = AudioSegment.from_file(args.file_path)
        spacer_milli = 2000
        spacer = AudioSegment.silent(duration=spacer_milli)
        audio = spacer.append(audio, crossfade=0)
        audio.export(args.file_path, format='wav')

    diarization_pipeline = None
    segmentation_model = None
    if args.mode == 'auto':
        diarization_pipeline = load_pyannote_audio_pipeline(args.diaization_model, args.hf_token)
        segmentation_model = load_pyannote_audio_model(args.segmentation_model, args.hf_token)
    elif args.mode == 'segment and transcribe':
        segmentation_model = load_pyannote_audio_model(args.segmentation_model, args.hf_token)
    elif args.mode == 'diarize':
        diarization_pipeline = load_pyannote_audio_pipeline(args.diaization_model, args.hf_token)

    if args.mode == 'diarize' or args.mode == 'auto':
        diarization = diarize_audio_file(args.file_path, diarization_pipeline)
        diarization_groups = group_diarization(diarization)
        gidx = segment_file_by_diargroup(args.file_path, os.path.join(output_path, 'wavs'), diarization_groups, original_file_path)
        with open(os.path.join(output_path, 'train.txt'), 'w') as f:
            f.write('')
        with open(os.path.join(output_path, 'valid.txt'), 'w') as f:
            f.write('')
        base_name = os.path.splitext(os.path.basename(original_file_path))[0]
        for idx, g in enumerate(diarization_groups):
            if idx < len(diarization_groups) * (1 - args.valid_ratio):
                with open(os.path.join(output_path, 'train.txt'), 'a') as f:
                    f.write(f'wavs/{base_name}-{idx}.wav\n')
            else:
                with open(os.path.join(output_path, 'valid.txt'), 'a') as f:
                    f.write(f'wavs/{base_name}-{idx}.wav\n')

    if args.mode == 'segment and transcribe' or args.mode == 'auto':
        segmentation = segment_audio_file(args.file_path, segmentation_model, args.seg_onset, args.seg_offset, args.seg_min_duration, args.seg_min_duration_off)
        segmentation_groups = group_segmentation(segmentation)
        gidx = segment_file_by_diargroup(args.file_path, os.path.join(output_path, 'wavs'), segmentation_groups, original_file_path)
        with open(os.path.join(output_path, 'train.txt'), 'w') as f:
            f.write('')
        with open(os.path.join(output_path, 'valid.txt'), 'w') as f:
            f.write('')
        base_name = os.path.splitext(os.path.basename(original_file_path))[0]
        for idx, g in enumerate(segmentation_groups):
            if idx < len(segmentation_groups) * (1 - args.valid_ratio):
                with open(os.path.join(output_path, 'train.txt'), 'a') as f:
                    f.write(f'wavs/{base_name}-{idx}.wav\n')
            else:
                with open(os.path.join(output_path, 'valid.txt'), 'a') as f:
                    f.write(f'wavs/{base_name}-{idx}.wav\n')

    if args.mode == 'transcribe' or args.mode == 'auto':
        pipe = init_transcribe_pipeline(args.whisper_model, 0 if args.device == 'cuda' else -1)
        for idx in range(gidx + 1):
            file_path = os.path.join(output_path, 'wavs', f'{os.path.splitext(os.path.basename(original_file_path))[0]}-{idx}.wav')
            transcription = transcribe_audio(file_path, pipe)
            add_to_textfile(os.path.join(output_path, 'train.txt'), f'{os.path.basename(file_path)}|{transcription}\n')

    cprint(figlet_format('DONE', font='starwars'), 'green', 'on_blue', attrs=['bold'])
