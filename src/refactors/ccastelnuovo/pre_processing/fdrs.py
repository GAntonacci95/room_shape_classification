import numpy as np
# numba dovrebbe accrescere le prestazioni, ma se usato male fa il contrario.
from numba import jit


# lo script corrente salva gli indici delle fdr, con un altro script si può facilmente
# ritagliare i pezzi di segnale indicati

# @jit(nopython=True) evita warning per tipi python schedulati alla deprecazione?

# TODO: introdurre migliorie sulle tipizzazioni di jit

# USATO NEL "MAIN LOCALE"
# per controllare se due fdr si intersecano e, nel caso, prendere quella più lunga


@jit    #(int64[:](int64[:], int64[:]))
def control_longer_fdr(indexes_first_fdr, indexes_second_fdr):
    if (indexes_first_fdr[1] < indexes_second_fdr[0]) or (indexes_first_fdr[0] > indexes_second_fdr[1]):
        return indexes_first_fdr
    else:
        if indexes_first_fdr[1] - indexes_first_fdr[0] > indexes_second_fdr[1] - indexes_second_fdr[0]:
            return indexes_first_fdr
        else:
            return indexes_second_fdr


# USATO IN find_fdr
# controllo a valle della fdr per vedere di riallacciare una fdr divisa in due da un frame
@jit    #(int64[:](int64, int64, int64, int64, float32[:]))
def following_control(win_len, win_offset, first_index, last_index, signal):
    temp_first_index = first_index
    temp_last_index = last_index
    frame = np.sum(np.square(signal[first_index:last_index]))
    # controllo un tot di finestre avanti
    tot = 3
    # caso particolare della fine del segnale, in cui il rumore ha più effetto sul segnale
    if last_index > 8 / 10 * len(signal):
        tot = len(signal)
    # controllo per non essere oltre la fine del segnale
    for i in range(tot):
        if tot * win_offset > (len(signal) - win_len):
            tot -= 1
    # faccio il controllo per ogni finestra selezionata a valle
    for i in range(tot):
        following_frame = signal[first_index + (i + 1) * win_offset:last_index + (i + 1) * win_offset]
        following_frame = np.sum(np.square(following_frame))
        if following_frame < frame:
            temp_last_index = last_index + (i + 1) * win_offset
    return [temp_first_index, temp_last_index]


# USATO IN find_fdr
# controllo a monte di una regione per riallacciare una fdr divisa in due da un frame
@jit    #(int64[:](int64, int64, int64, float32[:]))
def previous_control(win_offset, first_index, last_index, signal):
    temp_first_index = first_index
    temp_last_index = last_index
    frame = np.sum(np.square(signal[first_index:last_index]))
    tot = 6
    # caso particolare alla fine del segnale
    if last_index > 8 / 10 * len(signal):
        # prendo tutto il segnale prima. Alla fine del segnale il rumore si fa sentire di più
        tot = int(round((first_index - 7 / 10 * len(signal)) / win_offset))
    # controllo che non parta da dopo l'inizio della fdr
    for i in range(tot):
        if first_index < win_offset * (tot - i):
            tot -= (i + 1)
    # controllo che i frame precedenti siano maggiori di quello corrente
    for i in range(tot):
        previous_frame = signal[first_index - (i + 1) * win_offset:last_index - (i + 1) * win_offset]
        previous_frame = np.sum(np.square(previous_frame))
        if previous_frame > frame:
            temp_first_index = first_index - (i + 1) * win_offset
    return [temp_first_index, temp_last_index]


# USATO IN find_fdr
# per evitare di avere un collo di bottiglia verso la fine del segnale posso mettere
# un controllo che vede che siamo alla fine del segnale e prende tutto
# controllo all'interno della stessa fdr se ci sono porzioni crescenti e le elimino
@jit    #(int64[:](int64, int64, float32[:]))
def crescent_control(first_index, last_index, signal):
    i = 0
    frame = np.square(signal[first_index:last_index])
    # controllo di cui sopra
    if last_index > int(round(95 / 100 * len(signal))):
        frame = np.square(signal[first_index:int(round(last_index / 3 * 2))])
    aggressivity = int(round(10 / 100 * len(frame)))

    # per ogni sample del frame
    while i < (len(frame)):
        counter = 0
        # vado a controllare un numero di samples precedenti
        for j in range(min(aggressivity, i)):
            if frame[i] > frame[i - j]:
                counter += 1
        # se il sample è maggiore di tot precedenti, taglio il mio segnale
        if counter >= int(round(min(aggressivity - 1, i))):
            if i > int(round(len(frame) / 2)):
                frame = frame[:i]
                last_index -= (last_index - i - first_index)
            else:
                frame = frame[i:]
                first_index += i
                i = 0
        i += 1
    if last_index > int(round(95 / 100 * len(signal))):
        last_index = len(signal) - 1
    return [first_index, last_index]


# refactor wrt Castelnuovo
@jit    #(int64[:,:](int64, float32[:]))
def find_fdr(win_len, signal):
    fdrs = []
    num_min_frames = 5
    window_len = win_len
    window_overlap = round(window_len / 4)

    k = 0
    frames = []
    # finestro il segnale
    while k < (len(signal) - window_len):
        frames.append(signal[k:k + window_len])
        k += (window_len - window_overlap)
    previous_energy = np.sum(np.square(frames[0])) + 1
    frame_count = 0
    first_index = []
    # controllo che ci siano sequenze di tot frame con energia decrescente
    for m in range(len(frames)):
        frames[m] = np.sum(np.square(frames[m]))
        if frames[m] <= previous_energy:
            frame_count += 1
            if len(first_index) == 0:
                first_index.append(m)
        if frames[m] > previous_energy:
            # se sono alla fine della sequenza faccio i vari controlli e salvo
            if frame_count >= num_min_frames:
                first_term = (window_len - window_overlap) * first_index[0]
                last_term = (window_len - window_overlap) * m
                f_indexes = following_control(win_len, (win_len - window_overlap), first_term, last_term, signal)
                p_indexes = previous_control((win_len - window_overlap), first_term, last_term, signal)
                fdrs.append(crescent_control(p_indexes[0], f_indexes[1], signal))
            if len(first_index) != 0:
                first_index.pop(0)
            frame_count = 0
        previous_energy = frames[m]
    return fdrs


# refactor wrt Castelnuovo
# per l'ottimizzazione jit, band sia vettore float
def entry_point(signal, fs):
    # import framework.extension.scipy as spe
    # from obspy.signal.filter import envelope
    incipit = signal
    # incipit = envelope(incipit)
    # incipit = spe.lp_filter(incipit.reshape((1, -1)), 50, fs).reshape((-1))
    win_lens = range(300, 1001, 100)    # da 200 a 300 come min... accettabile con fs=16KHz
    ret = []
    variable_win_fdrs = [find_fdr(win_len, incipit) for win_len in win_lens]

    final_fdrs = []
    # vado a prendere le fdr della stanza per la lunghezza della finestra minore
    # [i][j]: i="indice win_len", j="indice fdr"
    for y in range(len(variable_win_fdrs[0])):
        temp_fdr = variable_win_fdrs[0][y]
        # vado a confrontare con le fdr delle varie lunghezze di finestra
        for i in range(len(variable_win_fdrs)):
            for j in range(len(variable_win_fdrs[i])):
                temp_fdr = control_longer_fdr(temp_fdr, variable_win_fdrs[i][j])
        if temp_fdr not in final_fdrs:
            final_fdrs.append(temp_fdr)

    for fdr in final_fdrs:
        ret.append({
            "begin_at": fdr[0],
            "end_at": fdr[1],
            "slice": signal[fdr[0]:fdr[1]]
        })

    return ret
