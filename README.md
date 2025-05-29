## 실행 명령어
만약 실행하고 싶은 실험이 1-1 실험이라면, 
``` bash
./experiment_scripts/1-1/train_1-1.sh

or

bash train.sh 1-1
# bash /experiment_scripts/1-1/train_1-1.sh
```
를 실행하면 된다.


## Auto make scripts
코드 복사 용도
```bash
python auto_make_scripts.py 1-1 1-2
```

## cut_screen
만든 screen 다 지우기
```bash
bash cut_screen.sh
```

## cut_screen
모든 experiment_output_image에 experiment_scripts에 작성한 readme 파일 추가
```bash
bash readme.sh
```