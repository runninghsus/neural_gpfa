from utilities.load_bsoid import *


class bsoid_behavior:

    def __init__(self):
        self.working_dir, self.prefix = query_workspace()
        self.f_index = []
        self.new_predictions = []

    def find_files(self):
        _, _, filenames, _, self.new_predictions = load_predictions(self.working_dir, self.prefix)
        f_partition = [filenames[i].rpartition('/')[-1] for i in range(len(filenames))]
        file4vid = st.selectbox('Which file corresponds to the the Kilosort2 analysis?',
                                f_partition, index=0)
        self.f_index = f_partition.index(file4vid)

    def main(self):
        self.find_files()
        return self.f_index, self.new_predictions
