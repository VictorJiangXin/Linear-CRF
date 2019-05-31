from crf import LinearCRF


def main():
    model = LinearCRF()
    model.train('../data/train.data')


if __name__ == '__main__':
    main()