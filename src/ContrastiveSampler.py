from collections import defaultdict
import random


random.seed(4)

class ContrastiveSampler:
    def __init__(self, train_data, pos_samples=5, neg_samples=5):
        self.train_data = train_data
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

    def sample_data(self):
        self.__sample_idxs()
        formated_output = self.__format_idxs()
        return formated_output

    def __sample_idxs(self):
        self.sampled_idxs = defaultdict(list)
        class_idxs, classes_list = self.__get_class_idxs()
        for idx, target in enumerate(self.train_data.targets):
            # select random idxs of positive class
            pos_sample_idxs = random.sample(
                class_idxs[target.item()], k=self.pos_samples)
            self.sampled_idxs[idx].extend(
                [(pos_sample_idx, 1) for pos_sample_idx in pos_sample_idxs])

            # select random idxs of negative classes
            neg_classes = random.choices(
                [x for x in classes_list if x != target.item()], k=self.neg_samples)
            for neg_class in neg_classes:
                neg_sample_idx = random.sample(class_idxs[neg_class], k=1)[0]
                self.sampled_idxs[idx].append((neg_sample_idx, 0))

        return

    def __get_class_idxs(self):
        class_idxs = defaultdict(list)
        for idx, target in enumerate(self.train_data.targets):
            class_idxs[target.item()].append(idx)
        return class_idxs, list(class_idxs.keys())

    def __format_idxs(self):
        formated_output = list()
        for anchor_id, samples in self.sampled_idxs.items():
            for sample_id, is_pos in samples:
                formated_output.append([anchor_id, sample_id, is_pos])
        return formated_output


if __name__ == "__main__":
    pass



