from data import Data

def main():
	data = Data('miniproject_data/norb_binary.mat')
	data.importDataFromMat()
	data.normalizeDataset()

main()

