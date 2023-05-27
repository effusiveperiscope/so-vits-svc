import os
import argparse
import subprocess
from pathlib import Path

def default_checkpoint_path(ckpt_dir = "chkpt/sovits5.0"):
    if os.path.exists(ckpt_dir) and len(os.listdir(ckpt_dir)):
        return os.path.join(ckpt_dir, (sorted(os.listdir(ckpt_dir))[-1]))
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--sample_rate', type=int, default=32000,
        help='sample rate to use for model')
    parser.add_argument('-n', '--name', type=str, default='sovits5.0',
        help='name of model for logging,s aving checkpoint')
    args = parser.parse_args()

    os.environ['PYTHONPATH'] = os.getcwd()
    assert os.path.exists(os.path.join(os.environ['PYTHONPATH'],
        'requirements.txt'))

    print('note: errors are not trivial and may disturb future runs if'
          ' unresolved')
    if not os.path.exists('data_svc/waves-32k'):
        subprocess.check_call(['python','prepare/preprocess_a.py',
            '-w','data_svc_raw','-o','data_svc/waves-32k',
            '-s',str(args.sample_rate)], env=os.environ)
    if not os.path.exists('data_svc/pitch'):
        subprocess.check_call(['python','prepare/preprocess_f0.py',
            '-w','data_svc/waves-32k',
            '-p','data_svc/pitch'], env=os.environ)
    if not os.path.exists('data_svc/whisper'):
        subprocess.check_call(['python','prepare/preprocess_ppg.py',
            '-w','data_svc/waves-32k',
            '-p','data_svc/whisper'], env=os.environ)
    if not os.path.exists('data_svc/speaker'):
        subprocess.check_call(['python','prepare/preprocess_speaker.py',
            'data_svc/waves-32k',
            'data_svc/speaker'], env=os.environ)
    if not os.path.exists('data_svc/singer'):
        subprocess.check_call(['python','prepare/preprocess_speaker_ave.py',
            'data_svc/speaker',
            'data_svc/singer'], env=os.environ)
    if not os.path.exists('data_svc/specs'):
        subprocess.check_call(['python','prepare/preprocess_spec.py',
            '-w','data_svc/waves-32k',
            '-s','data_svc/specs'], env=os.environ)
    subprocess.check_call(['python','prepare/preprocess_train.py'],
        env=os.environ)
    subprocess.check_call(['python','prepare/preprocess_zzz.py'],
        env=os.environ)
    warmstart = default_checkpoint_path(os.path.join('chkpt',args.name))
    if warmstart is not None:
        subprocess.check_call(['python','svc_trainer.py',
            '-c','configs/base.yaml',
            '-p',warmstart,
            '-n',args.name], env=os.environ)
    else:
        subprocess.check_call(['python','svc_trainer.py',
            '-c','configs/base.yaml',
            '-n',args.name], env=os.environ)
