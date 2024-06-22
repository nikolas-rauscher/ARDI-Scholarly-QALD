import os
import sys
sys.path.append('./src')
def main():
    os.system("python3 src/models/prepare_prompts_context.py")
    os.system("python3 src/models/zero_shot_prompting2.py")

if __name__ == '__main__':
    main()