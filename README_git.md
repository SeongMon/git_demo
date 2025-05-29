# 🌿 Git 브랜치 명령어 정리

실제로 돌리면서 익혀야 한다.

## ✅ 브랜치 목록 보기
```bash
git branch
```
- 현재 로컬 브랜치 목록 확인
- 현재 선택된 브랜치 앞에 `*` 표시됨

---

## ✅ 새 브랜치 생성 & 이동
```bash
git checkout -b 브랜치이름
```
예시:
```bash
git checkout -b feature-new-loss
```

---

## ✅ 브랜치 전환
```bash
git checkout 브랜치이름
```
예시:
```bash
git checkout main
```

---

## ✅ 로컬 브랜치 삭제
```bash
git branch -d 브랜치이름      # 병합된 브랜치만 삭제
git branch -D 브랜치이름      # 병합 안됐어도 강제 삭제
```
예시:
```bash
git branch -d feature-new-loss
```

---

## ✅ 원격 브랜치 삭제 (GitHub에서도 삭제됨)
```bash
git push origin --delete 브랜치이름
```
예시:
```bash
git push origin --delete feature-new-loss
```

---

## ✅ 브랜치 푸시 (GitHub로 올리기)
```bash
git push origin 브랜치이름
```
예시:
```bash
git push origin feature-new-loss
```

---

## ✅ `main` 브랜치 최신 상태로 갱신
```bash
git checkout main
git pull origin main
```

---

## ✅ PR 전용 브랜치 만들기 (main 기반)
```bash
git checkout main
git pull origin main
git checkout -b feature-새작업
```

---
