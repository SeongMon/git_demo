import os
import shutil
import argparse
import sys

### python auto_make_scripts.py 1-1 1-2
### 1-1 폴더를 복사하여 1-2 폴더를 만들고 안에 파일 내용을 복사하고 .sh파일에서 export EXP_NUM 부분만 바꿔줌
### 수정: 이미 version_new (예: 1-2) 폴더가 존재하면 스크립트를 중단하고 에러 메시지 출력

def duplicate_experiment_with_specific_changes(version_old, version_new):
    base_dir = "experiment_scripts"
    src_dir = os.path.join(base_dir, version_old)
    dst_dir = os.path.join(base_dir, version_new)

    # 1. Check if source directory exists
    if not os.path.exists(src_dir):
        print(f"❌ Source directory does not exist: {src_dir}")
        sys.exit(1)

    # 2. Check if destination directory already exists
    if os.path.exists(dst_dir):
        print(f"❌ Error: Destination directory already exists: {dst_dir}")
        print(f"   Please remove the existing directory or choose a different 'version_new'.")
        sys.exit(1)

    # 3. Create destination directory
    try:
        os.makedirs(dst_dir)
        print(f"✅ Destination directory created: {dst_dir}")
    except OSError as e:
        print(f"❌ Error creating destination directory {dst_dir}: {e}")
        sys.exit(1)

    file_processed_count = 0
    try:
        for filename in os.listdir(src_dir):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            if os.path.isdir(src_path):
                if filename == "__pycache__":
                    print(f"🚫 Skipping __pycache__ directory: {src_path}")
                    continue
                try:
                    shutil.copytree(src_path, dst_path, ignore=shutil.ignore_patterns('__pycache__'))
                    print(f"📁 Copied directory: {src_path} → {dst_path}")
                except Exception as e:
                    print(f"⚠️ Error copying directory {src_path}: {e}")
                continue

            # ➡️ 파일인 경우
            new_filename = filename.replace(version_old, version_new)
            dst_path = os.path.join(dst_dir, new_filename)

            # 파일 읽기
            try:
                with open(src_path, 'r', encoding='utf-8') as src_file:
                    content = src_file.read()
            except Exception as e:
                print(f"⚠️ Error reading source file {src_path}: {e}")
                continue

            # 특정 파일 내용 수정
            if filename == f'train_{version_old}.sh':
                content = content.replace(f'export EXP_NUM="{version_old}"', f'export EXP_NUM="{version_new}"')
                content = content.replace(f'--exp_num {version_old}', f'--exp_num {version_new}')
            if filename == '@readme.txt':
                content = f'{version_new}'
            if filename == f'train_multi_{version_old}.sh':
                content = content.replace(f'export EXP_NUM="{version_old}"', f'export EXP_NUM="{version_new}"')
                content = content.replace(f'--exp_num {version_old}', f'--exp_num {version_new}')

            # 새 파일 쓰기
            try:
                with open(dst_path, 'w', encoding='utf-8') as dst_file:
                    dst_file.write(content)
                print(f"✅ Copied and modified: {src_path} → {dst_path}")
                file_processed_count += 1
            except Exception as e:
                print(f"⚠️ Error writing destination file {dst_path}: {e}")

    except Exception as e:
        print(f"❌ An error occurred during file processing: {e}")
        print(f"ℹ️ Rolling back: Removing created directory {dst_dir}")
        if os.path.exists(dst_dir):
            try:
                shutil.rmtree(dst_dir)
                print(f"✅ Directory {dst_dir} removed.")
            except Exception as rm_e:
                print(f"⚠️ Failed to remove directory {dst_dir}: {rm_e}")
        sys.exit(1)

    print(f"\n✨ Successfully duplicated experiment {version_old} to {version_new}.")
    print(f"   Processed {file_processed_count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Duplicate experiment scripts with version change. Stops if the new version directory already exists."
    )
    parser.add_argument("version_old", type=str, help="The old version (e.g., 1-1)")
    parser.add_argument("version_new", type=str, help="The new version (e.g., 1-2)")

    args = parser.parse_args()

    duplicate_experiment_with_specific_changes(args.version_old, args.version_new)
