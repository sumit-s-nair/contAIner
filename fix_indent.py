lines = open('scripts/train_system2_grpo.py', 'r', encoding='utf-8').readlines()
for i in range(343, 601): # 0-indexed, so lines 344 to 601
    lines[i] = '    ' + lines[i]
with open('scripts/train_system2_grpo.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
