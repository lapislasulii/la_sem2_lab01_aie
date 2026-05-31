# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def _core_slice(core: DenseTensor, i: int) -> DenseTensor:
    """Возвращает срез ядра G_k[i] — матрицу (r_{k-1}, r_k)."""
    rl, n, rr = core.shape
    data = [0.0] * (rl * rr)
    for a in range(rl):
        for b in range(rr):
            data[a * rr + b] = core.data[a * n * rr + i * rr + b]
    return DenseTensor((rl, rr), data=data)


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    d = tt1.order
    if d == 1:
        return TTTensor([backend.add(tt1.cores[0], tt2.cores[0])])

    new_cores: list[DenseTensor] = []
    for k in range(d):
        A = tt1.cores[k]
        B = tt2.cores[k]
        al, n, ar = A.shape
        bl, _, br = B.shape

        if k == 0:
            rl, rr = 1, ar + br
        elif k == d - 1:
            rl, rr = al + bl, 1
        else:
            rl, rr = al + bl, ar + br

        loff = 0 if k == 0 else al
        roff = 0 if k == d - 1 else ar

        data = [0.0] * (rl * n * rr)
        for a in range(al):
            for i in range(n):
                for c in range(ar):
                    data[a * n * rr + i * rr + c] = A.data[a * n * ar + i * ar + c]
        for a in range(bl):
            for i in range(n):
                for c in range(br):
                    data[(a + loff) * n * rr + i * rr + (c + roff)] = \
                        B.data[a * n * br + i * br + c]

        new_cores.append(DenseTensor((rl, n, rr), data=data))

    return TTTensor(new_cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    new_cores = [core.copy() for core in tt.cores]
    new_cores[0] = backend.scale(new_cores[0], alpha)
    return TTTensor(new_cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    d = tt1.order
    new_cores: list[DenseTensor] = []
    for k in range(d):
        A = tt1.cores[k]
        B = tt2.cores[k]
        al, n, ar = A.shape
        bl, _, br = B.shape
        rl, rr = al * bl, ar * br

        data = [0.0] * (rl * n * rr)
        for i in range(n):
            for a in range(al):
                for c in range(ar):
                    va = A.data[a * n * ar + i * ar + c]
                    if va == 0.0:
                        continue
                    for ap in range(bl):
                        for cp in range(br):
                            p = a * bl + ap
                            q = c * br + cp
                            data[p * n * rr + i * rr + q] = \
                                va * B.data[ap * n * br + i * br + cp]

        new_cores.append(DenseTensor((rl, n, rr), data=data))

    return TTTensor(new_cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    d = tt1.order
    Z = None
    for k in range(d):
        A = tt1.cores[k]
        B = tt2.cores[k]
        n = A.shape[1]
        acc = None
        for i in range(n):
            At = backend.transpose(_core_slice(A, i))
            Bsl = _core_slice(B, i)
            if Z is None:
                term = backend.matmul(At, Bsl)
            else:
                term = backend.matmul(backend.matmul(At, Z), Bsl)
            acc = term if acc is None else backend.add(acc, term)
        Z = acc
    return Z.data[0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    return math.sqrt(max(tt_dot(tt, tt, backend), 0.0))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    aa = tt_dot(tt1, tt1, backend)
    ab = tt_dot(tt1, tt2, backend)
    bb = tt_dot(tt2, tt2, backend)
    return math.sqrt(max(aa - 2.0 * ab + bb, 0.0))