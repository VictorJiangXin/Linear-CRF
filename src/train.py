from crf import LinearCRF


def main():
    print("main start")
    model = LinearCRF()
    model.train('../data/msr_training.data')


if __name__ == '__main__':
    main()