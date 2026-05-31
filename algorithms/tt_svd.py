# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    shape = tensor.shape
    d = tensor.ndim

    if d == 1:
        return TTTensor([tensor.reshape((1, shape[0], 1))])

    norm_a = backend.norm(tensor)
    delta = (eps / math.sqrt(d - 1)) * norm_a if norm_a > 1e-30 else 0.0

    cores: list[DenseTensor] = []
    C = tensor.copy()
    r_prev = 1

    for k in range(d - 1):
        n_k = shape[k]
        ncols = C.size // (r_prev * n_k)
        matrix = C.reshape((r_prev * n_k, ncols))

        U, S, Vt = backend.svd(matrix)
        rk = _compute_truncated_rank(S, delta, max_rank)

        U_tr = _truncate_columns(U, rk, backend)
        cores.append(U_tr.reshape((r_prev, n_k, rk)))

        S_tr = _truncate_vector(S, rk, backend)
        Vt_tr = _truncate_rows(Vt, rk, backend)
        C = _multiply_diag_matrix(S_tr, Vt_tr, rk, backend)
        r_prev = rk

    cores.append(C.reshape((r_prev, shape[d - 1], 1)))
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    k = S.shape[0]
    if k == 0:
        return 1

    sigma1 = S.data[0]
    threshold = max(1e-12, 1e-8 * sigma1)

    r_hat = 0
    for j in range(k):
        if S.data[j] > threshold:
            r_hat += 1
        else:
            break

    rk = r_hat
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

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

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