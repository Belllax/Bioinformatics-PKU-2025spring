from ProfileHMM import *
from multiprocessing import Pool, TimeoutError, freeze_support
from sys import exit
import numpy as np
import time


np.set_printoptions(linewidth=14000000, precision=4, threshold=1000000)

def _plot(y):
    import matplotlib.pyplot as plt
    for data, c, marker in zip(y, 'ACGU', 'xo.v'):
        data = data[:100]
        plt.plot(np.arange(len(data)), data, label=c, marker=marker, ls='None')
    plt.xlabel('Match state number')
    plt.ylabel('Propabilities')
    plt.legend(loc='upper right')
    plt.show()

def read(rfile):
    MSA = []
    with rfile as train_data:
        for line in train_data.readlines():
            if line.startswith('>'):
                continue
            MSA.append(np.array(list(line.strip())))
    return np.array(MSA)

def testdata_iter(testdata):
    for line in testdata.readlines():
        if line.startswith('>'):
            continue
        yield line
    testdata.seek(0)


def save_hmm_text_output(HMM_MSA, filename="hmm_matrices.txt"):
    with open(filename, "w") as out:
        print("[Info] Writing emission matrix to:", out.name)
        print("Emission Matrix (Match States):", file=out)
        emi = HMM_MSA.emissions_from_M

        if isinstance(emi, dict):
            for state, probs in emi.items():
                print(f"{state}:\t" + "\t".join(f"{p:.4f}" for p in probs), file=out)
        else:
            try:
                import numpy as np
                emi = np.array(emi)
                for i, row in enumerate(emi):
                    print(f"M{i+1}:\t" + "\t".join(f"{p:.4f}" for p in row), file=out)
            except Exception as e:
                print("Could not print emissions:", e, file=out)

        print("\nTransition Matrix:", file=out)
        trans = HMM_MSA.transmissions
        if isinstance(trans, dict):
            for from_state, to_dict in trans.items():
                for to_state, prob in to_dict.items():
                    print(f"{from_state} -> {to_state}:\t{prob:.4f}", file=out)
        else:
            try:
                trans = np.array(trans)
                for i, row in enumerate(trans):
                    print("State {}:\t".format(i) + "\t".join(f"{p:.4f}" for p in row), file=out)
            except Exception as e:
                print("Could not print transitions:", e, file=out)

    print(f"[Success] Emission and transition matrices written to: {filename}")


def main():
    train, test, out = parseme()
    MSA = read(train)
    hmmprofile = HMM(MSA)
        
    save_hmm_text_output(hmmprofile)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i, base in enumerate(hmmprofile.MSAchar):
        plt.plot(hmmprofile.emissions_from_M[i], label=base, marker='o')
    plt.title('Emission Probabilities from Match States')
    plt.xlabel('Match State')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig("emissions_plot.png")
    print("[Info] Emission matrix plot saved as emissions_plot.png")

    plt.figure(figsize=(10, 6))
    plt.imshow(hmmprofile.transmissions, aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.title('Transition Probability Matrix (heatmap)')
    plt.xlabel('Match State')
    plt.ylabel('Transition Type')
    plt.yticks(ticks=np.arange(9), labels=[
        'M->M', 'M->D', 'M->I',
        'I->M', 'I->I', 'I->D',
        'D->M', 'D->D', 'D->I'])
    plt.tight_layout()
    plt.savefig("transitions_heatmap.png")
    print("Transition matrix heatmap saved as transitions_heatmap.png")


    index = []
    scores = []
    seq_start = [line[:30] for line in testdata_iter(test)]
    num_lines = sum(1 for line in test) // 2 - 1
    test.seek(0)

    with Pool() as pool:
        for i, score in enumerate(pool.imap(hmmprofile.viterbi, testdata_iter(test))):
            print('[{} of {}] in progress..'.format(i, num_lines), end='\r')
            index.append(i)
            scores.append(score)
        print()


    for i, start, score in zip(index, seq_start, scores):
        print("{}\t{}\t{}".format(i, start, score), file=out)

if __name__ == "__main__":
    start_time = time.time()
    
    main()

    end_time = time.time()
    print(f"\n⏱️ 程序运行时间：{end_time - start_time:.2f} 秒")