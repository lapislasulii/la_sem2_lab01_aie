# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]
    d = len(cores)

    for k in range(d - 1):
        rl, nk, rr = cores[k].shape
        mat = cores[k].reshape((rl * nk, rr))

        if rl * nk >= rr:
            Q, R = backend.qr(mat)
            new_rr = rr
        else:
            U, S, Vt = backend.svd(mat)
            new_rr = rl * nk
            Q = U
            R = backend.matmul(backend.diag(S), Vt)

        cores[k] = Q.reshape((rl, nk, new_rr))

        rl2, nk2, rr2 = cores[k + 1].shape
        nxt = cores[k + 1].reshape((rl2, nk2 * rr2))
        cores[k + 1] = backend.matmul(R, nxt).reshape((new_rr, nk2, rr2))

    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]
    d = len(cores)

    for k in range(d - 1, 0, -1):
        rl, nk, rr = cores[k].shape
        mat = cores[k].reshape((rl, nk * rr))

        if nk * rr >= rl:
            Qt, Rt = backend.qr(backend.transpose(mat))
            Q = backend.transpose(Qt)
            R = backend.transpose(Rt)
            new_rl = rl
        else:
            U, S, Vt = backend.svd(mat)
            new_rl = nk * rr
            Q = Vt
            R = backend.matmul(U, backend.diag(S))

        cores[k] = Q.reshape((new_rl, nk, rr))

        rl0, nk0, rr0 = cores[k - 1].shape
        prev = cores[k - 1].reshape((rl0 * nk0, rr0))
        cores[k - 1] = backend.matmul(prev, R).reshape((rl0, nk0, new_rl))

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    k = S.shape[0]
    if k == 0:
        return 0

    threshold = max(abs_tol, rel_tol * abs(S.data[0]))
    rank = 0
    for j in range(k):
        if abs(S.data[j]) > threshold:
            rank += 1
        else:
            break
    return rank


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
        rank:     длина диагонального вектора
        backend:  интерфейс backend
    """
    n = matrix.shape[1]
    data = [0.0] * (rank * n)
    for i in range(rank):
        factor = diag_vec.data[i]
        for j in range(n):
            data[i * n + j] = factor * matrix.data[i * n + j]
    return DenseTensor((rank, n), data=data)


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    m, n = matrix.shape
    data = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            data[i * n + j] = matrix.data[i * n + j] * diag_vec.data[j]
    return DenseTensor((m, n), data=data)