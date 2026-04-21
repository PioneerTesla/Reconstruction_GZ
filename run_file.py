"""
批量训练调度器 - 支持多任务队列后台运行

参数管理方式:
  - 基础参数: 直接编辑 config.json
  - 每个任务的独立参数: 编辑 jobs.json，未指定的字段自动使用 config.json 中的值

用法示例:
    # 单次运行（使用 config.json 默认值）
    python run_file.py

    # 先跑 Miss 再跑 Mix（自动排队，参数来自 config.json）
    python run_file.py --scene Miss Mix

    # 多 root × 多 scene 全组合
    python run_file.py --root dataset dataset_random_len --scene Miss Spurious Mix

    # 使用 jobs.json（每个任务可设置不同参数）
    python run_file.py --jobs jobs.json

    # 预览所有任务配置但不实际运行
    python run_file.py --jobs jobs.json --dry_run

    # 从上次中断的地方继续（跳过已完成的任务）
    python run_file.py --jobs jobs.json --resume

jobs.json 格式示例:
    [
        {"scene": "Miss",  "root": "dataset"},
        {"scene": "Mix",   "root": "dataset", "learning_rate": 0.0003},
        {"scene": "Spurious", "root": "dataset_random_len", "diff_steps": 60}
    ]
    每个条目的字段会覆盖 config.json 中的对应值，未指定的字段保持默认。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import itertools
import time
from datetime import datetime
from typing import List


CONFIG_PATH = 'config.json'
TRAIN_SCRIPT = 'train_pri.py'
PROGRESS_FILE = 'batch_progress.json'
BATCH_LOG_FILE = 'batch_runner.log'
DEFAULT_JOBS_FILE = 'jobs.json'          # ← 默认任务文件，直接点运行就会读取
LOG_DIR = 'logs'                          # ← 每个任务的独立日志目录


def log(msg: str, log_file: str = BATCH_LOG_FILE) -> None:
    """同时输出到控制台和日志文件"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{timestamp}] {msg}'
    print(line)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def load_base_config() -> dict:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(cfg: dict, path: str = CONFIG_PATH) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


def load_progress() -> dict:
    """加载进度文件，记录哪些任务已完成"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'completed': []}


def save_progress(progress: dict) -> None:
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=4, ensure_ascii=False)


def make_job_id(job_cfg: dict) -> str:
    """根据关键参数生成任务唯一标识"""
    # scene = job_cfg.get('scene', 'Unknown')
    # root = os.path.basename(job_cfg.get('root', 'dataset').rstrip('/\\'))
    model = job_cfg.get('model_name', 'DiffSeqPRI')
    return f'{model}'


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f'{h}h {m}m {s}s'
    elif m > 0:
        return f'{m}m {s}s'
    return f'{s}s'


def load_jobs_from_file(jobs_path: str, base_cfg: dict) -> List[dict]:
    """从 jobs.json 加载任务列表，每个任务合并 base_cfg"""
    with open(jobs_path, 'r', encoding='utf-8') as f:
        jobs_list = json.load(f)

    jobs = []
    for entry in jobs_list:
        run_cfg = base_cfg.copy()
        run_cfg.update(entry)
        # 自动生成 model_name（如果未指定）
        if 'model_name' not in entry:
            scene = run_cfg.get('scene', 'Unknown')
            safe_root = os.path.basename(run_cfg.get('root', 'dataset').rstrip('/\\'))
            # run_cfg['model_name'] = f'DiffSeqPRI_{scene}_{safe_root}'
        jobs.append(run_cfg)
    return jobs


def build_jobs_from_cli(args, base_cfg: dict) -> List[dict]:
    """从 CLI 参数构建任务列表（root × scene 笛卡尔积）"""
    roots: List[str] = args.root if args.root else [base_cfg.get('root', 'dataset')]
    scenes: List[str] = args.scene if args.scene else [base_cfg.get('scene', 'Miss')]

    jobs = []
    for root, scene in itertools.product(roots, scenes):
        run_cfg = base_cfg.copy()
        run_cfg['root'] = root
        run_cfg['scene'] = scene
        safe_root = os.path.basename(root.rstrip('/\\')) or root
        # run_cfg['model_name'] = f'DiffSeqPRI_{scene}_{safe_root}'
        jobs.append(run_cfg)
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='批量训练调度器: 支持多任务队列后台顺序运行',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--jobs', type=str, default=None,
                        help='任务队列 JSON 文件路径。例: python run_file.py --jobs jobs.json')
    parser.add_argument('--root', nargs='+', default=None,
                        help='数据集根目录（可多个）。例: --root dataset dataset_random_len')
    parser.add_argument('--scene', nargs='+', default=None,
                        help='场景类型（可多个）。例: --scene Miss Spurious Mix')
    parser.add_argument('--dry_run', action='store_true',
                        help='仅预览所有任务配置，不实际运行')
    parser.add_argument('--resume', action='store_true',
                        help='断点续跑: 跳过已完成的任务')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    base_cfg = load_base_config()

    # 构建任务列表
    # 优先级: --jobs 指定文件 > 默认 jobs.json（如果存在）> --root/--scene 组合 > config.json 单次运行
    jobs_path = args.jobs
    if not jobs_path and os.path.isfile(DEFAULT_JOBS_FILE):
        jobs_path = DEFAULT_JOBS_FILE

    if jobs_path:
        if not os.path.isfile(jobs_path):
            print(f'[错误] 文件不存在: {jobs_path}')
            sys.exit(1)
        jobs = load_jobs_from_file(jobs_path, base_cfg)
        log(f'从 {jobs_path} 加载了 {len(jobs)} 个任务')
    else:
        jobs = build_jobs_from_cli(args, base_cfg)

    if not jobs:
        print('没有任务需要运行。')
        return

    # 加载进度（用于 --resume）
    progress = load_progress() if args.resume else {'completed': []}

    # 打印任务队列
    log(f'{"="*60}')
    log(f'批量训练调度器: 共 {len(jobs)} 个任务')
    log(f'{"="*60}')
    for i, job in enumerate(jobs):
        job_id = make_job_id(job)
        status = '✓ 已完成' if job_id in progress['completed'] else '待运行'
        log(f'  [{i+1}/{len(jobs)}] {job.get("model_name", "?")} | '
            f'scene={job.get("scene")} | root={job.get("root")} | {status}')
    log('')

    # 保存原始 config 用于最后恢复
    original_cfg = base_cfg.copy()
    total_start = time.time()
    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, job in enumerate(jobs):
        job_id = make_job_id(job)

        # 断点续跑: 跳过已完成的任务
        if args.resume and job_id in progress['completed']:
            log(f'[跳过] [{i+1}/{len(jobs)}] {job_id} (已完成)')
            skip_count += 1
            continue

        log(f'{"="*60}')
        log(f'[{i+1}/{len(jobs)}] 开始: {job.get("model_name")} | '
            f'scene={job.get("scene")} | root={job.get("root")}')

        # 显示与 base_cfg 不同的参数
        diff_params = {k: v for k, v in job.items() if base_cfg.get(k) != v}
        if diff_params:
            log(f'  覆盖参数: {json.dumps(diff_params, ensure_ascii=False)}')
        log(f'{"="*60}')

        if args.dry_run:
            print(json.dumps(job, indent=4, ensure_ascii=False))
            print()
            continue

        # 写入 config 并运行训练
        save_config(job)
        job_start = time.time()

        # 每个任务的独立日志文件
        os.makedirs(LOG_DIR, exist_ok=True)
        job_log_path = os.path.join(LOG_DIR, f'{job.get("model_name", "job")}.log')
        log(f'  任务日志: {job_log_path}')

        with open(job_log_path, 'w', encoding='utf-8') as job_log:
            # 写入任务配置头
            job_log.write(f'任务: {job_id}\n')
            job_log.write(f'开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            job_log.write(f'配置: {json.dumps(job, indent=2, ensure_ascii=False)}\n')
            job_log.write('=' * 60 + '\n')
            job_log.flush()

            # 启动子进程，实时输出到控制台和日志文件
            proc = subprocess.Popen(
                [sys.executable, '-u', TRAIN_SCRIPT],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            for raw_line in proc.stdout:
                line = raw_line.decode('utf-8', errors='replace')
                sys.stdout.write(line)
                job_log.write(line)
                job_log.flush()
            proc.wait()

            # 写入任务结尾
            job_log.write('\n' + '=' * 60 + '\n')
            job_log.write(f'结束时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            job_log.write(f'退出码: {proc.returncode}\n')

        job_duration = time.time() - job_start

        if proc.returncode != 0:
            log(f'[失败] [{i+1}/{len(jobs)}] {job_id} 退出码={proc.returncode} '
                f'(耗时 {format_duration(job_duration)})')
            fail_count += 1
        else:
            log(f'[完成] [{i+1}/{len(jobs)}] {job_id} '
                f'(耗时 {format_duration(job_duration)})')
            success_count += 1
            # 记录完成进度
            progress['completed'].append(job_id)
            save_progress(progress)

        # 显示剩余任务数
        remaining = len(jobs) - (i + 1)
        if remaining > 0:
            log(f'  剩余 {remaining} 个任务...')
        log('')

    # 恢复原始 config
    save_config(original_cfg)

    total_duration = time.time() - total_start
    log(f'{"="*60}')
    log(f'全部任务结束 | 成功: {success_count} | 失败: {fail_count} | 跳过: {skip_count} | '
        f'总耗时: {format_duration(total_duration)}')
    log(f'config.json 已恢复为原始配置')
    log(f'{"="*60}')


if __name__ == '__main__':
    main()
