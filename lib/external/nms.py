import numpy as np

def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    N = dets.shape[0]

    for i in range(N):
        maxscore = dets[i, 4]
        maxpos = i

        for pos in range(i + 1, N):
            if dets[pos, 4] > maxscore:
                maxscore = dets[pos, 4]
                maxpos = pos

        dets[[i, maxpos]] = dets[[maxpos, i]]

        tx1, ty1, tx2, ty2, ts = dets[i]

        pos = i + 1
        while pos < N:
            x1, y1, x2, y2, s = dets[pos]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = min(tx2, x2) - max(tx1, x1) + 1

            if iw > 0:
                ih = min(ty2, y2) - max(ty1, y1) + 1

                if ih > 0:
                    ua = (tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih
                    ov = iw * ih / ua

                    if method == 1:
                        weight = 1 - ov if ov > Nt else 1
                    elif method == 2:
                        weight = np.exp(-(ov * ov) / sigma)
                    else:
                        weight = 0 if ov > Nt else 1

                    dets[pos, 4] *= weight

                    if dets[pos, 4] < threshold:
                        dets[pos] = dets[N - 1]
                        N -= 1
                        pos -= 1

            pos += 1

    return list(range(N))