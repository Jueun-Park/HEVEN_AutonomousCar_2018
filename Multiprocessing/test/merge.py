from multiprocessing import Pool

text = "test"
def harvester(text, case):
    X = case[0]
    return text+ str(X)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=6)
    case = RAW_DATASET
    pool.map(harvester(text,case),case, 1)
    pool.close()
    pool.join()