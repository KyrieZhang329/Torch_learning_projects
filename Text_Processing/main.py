import os
import sys
import train
import generate

def main():
    print("运行训练")
    train.train()
    print("运行生成")
    generate.main()


if __name__ == '__main__':
    main()
