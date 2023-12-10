import streamlit as st


def run():


    imports = """from numba import cuda
import cupy as cp
import numpy as np"""


    groupby_function = """
    @cuda.jit
    def groupby(A,res):
        "A is gxnxm and we want to add all nxm in g"
        i = cuda.threadIdx.x
        j = cuda.threadIdx.y
        b = cuda.blockIdx.y
        cuda.atomic.add(res,b, A[b,i,j])

    """

    create_data = """A = cp.random.randint(1,10,(2,8,8))
    C = cp.zeros(2)"""

    function_call = """
groupby[(1,2), (8,8)](A,C) # mental model: 2 blocks for 2 levels in A and aggregate value of each element of A in that block using a single thread
print(C)
"""

    general_structure = """
    @cuda.jit
    def groupby(A,res):
        "A is gxnxm and we want to add all nxm in g"
        i = cuda.threadIdx.x
        j = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        bz = cuda.blockIdx.z
        tpbx = TPB[0]
        tpby = TPB[1]
        cuda.atomic.add(res,bz, A[bz, (bx*tpbx+i), (by*tpby+j)])

    """

    higher_volume = """
    A = cp.random.randint(1,10,(2,512,512))
    C = cp.zeros(2)
    """

    function_call = """TPB = (32,32)
    BPG = (16,16,2)
    groupby[BPG, TPB](A,C) # mental model: 2 blocks for 2 levels in A and aggregate value of each element of A in that block using a single thread

    print(C)
    """

