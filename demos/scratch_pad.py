A0 = [1, 11, 2, 10, 4, 5, 2, 1]

def longest_inc_seq(A):
    if len(A) == 1:
        return A
    max_len = 0
    max_seq = []
    for n, a in enumerate(A):
        left_seq = []
        right_seq = []
        for la in A[:n]:
            if la < a:
                left_seq.append(la)
        for ra in A[n+1:]:
            if ra > a:
                right_seq.append(ra)

        l_ss = longest_inc_seq(left_seq)
        r_ss = longest_inc_seq(right_seq)
        subseq = l_ss + [a] + r_ss
        ss_len = len(subseq)
    
        if ss_len > max_len:
            max_len = ss_len
            max_seq = subseq
    return max_seq

def longest_dec_seq(A):
    A_inv = []
    for a in A:
        A_inv.append(-a)
    seq = []
    for a in longest_inc_seq(A_inv):
        seq.append(-a)
    return seq

def longest_hump_seq(A):
    max_len = 0
    for n, a in enumerate(A):
        left_A = []
        right_A = []
        for x in A[:n]:
            if x < a:
                left_A.append(x)
        for x in A[n+1:]:
            if x < a:
                right_A.append(x)
        seq = longest_inc_seq(left_A) + [a] + longest_dec_seq(right_A)
        if len(seq) > max_len:
            max_len = len(seq)
            max_seq = seq
    return max_len

