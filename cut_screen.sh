screen -ls | grep -o '[0-9]*\.' | awk '{print substr($0, 1, length($0)-1)}' | xargs -I {} screen -S {} -X quit
