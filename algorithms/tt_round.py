# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    d = tt.order
    if d == 1:
        return tt.copy()

    cores = right_canonicalize(tt, backend).cores

    delta = eps * backend.norm(cores[0]) / math.sqrt(d - 1)

    for k in range(d - 1):
        rl, n, rr = cores[k].shape
        matrix = cores[k].reshape((rl * n, rr))

        U, S, Vt = backend.svd(matrix)
        r_new = _compute_rank(S, delta, max_rank)

        U_tr = _truncate_columns(U, r_new, backend)
        cores[k] = U_tr.reshape((rl, n, r_new))

        S_tr = _truncate_vector(S, r_new, backend)
        Vt_tr = _truncate_rows(Vt, r_new, backend)
        M = _multiply_diag_matrix(S_tr, Vt_tr, r_new, backend)

        rl2, n2, rr2 = cores[k + 1].shape
        nxt = cores[k + 1].reshape((rl2, n2 * rr2))
        cores[k + 1] = backend.matmul(M, nxt).reshape((r_new, n2, rr2))

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    k = S.shape[0]
    if k == 0:
        return 1

    rk = k
    if delta > 0.0:
        tail = 0.0
        while rk > 1:
            s = S.data[rk - 1]
            if tail + s * s <= delta * delta:
                tail += s * s
                rk -= 1
            else:
                break

    if max_rank is not None:
        rk = min(rk, max_rank)
    return max(1, rk)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    m, n = matrix.shape
    data = [0.0] * (m * rank)
    for i in range(m):
        for j in range(rank):
            data[i * rank + j] = matrix.data[i * n + j]
    return DenseTensor((m, rank), data=data)


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    n = matrix.shape[1]
    return DenseTensor((rank, n), data=matrix.data[:rank * n])


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    return DenseTensor((rank,), data=vector.data[:rank])


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    n = matrix.shape[1]
    data = [0.0] * (rank * n)
    for i in range(rank):
        factor = diag_vec.data[i]
        for j in range(n):
            data[i * n + j] = factor * matrix.data[i * n + j]
    return DenseTensor((rank, n), data=data)