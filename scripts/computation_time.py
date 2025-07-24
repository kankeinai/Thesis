import numpy as np

full_time = {
    'linear': {
        'deeponet': sum(np.load("computation_time/linear/deeponet/unsupervised/time_epoch_1800.npy"))/60,
        'fno': sum(np.load("computation_time/linear/fno/unsupervised/time_epoch_180.npy"))/60,
        'lno': sum(np.load("computation_time/linear/lno/unsupervised/time_epoch_670.npy"))/60,
    },
    'nonlinear': {
        'deeponet': sum(np.load("computation_time/nonlinear/deeponet/unsupervised/time_epoch_1500.npy"))/60,
        'fno': sum(np.load("computation_time/nonlinear/fno/unsupervised/time_epoch_620.npy"))/60,
        'lno': sum(np.load("computation_time/nonlinear/lno/unsupervised/time_epoch_370.npy"))/60,
    },
    'oscillatory': {
        'deeponet': sum(np.load("computation_time/oscillatory/deeponet/unsupervised/time_epoch_1500.npy"))/60,
        'fno': sum(np.load("computation_time/oscillatory/fno/unsupervised/time_epoch_770.npy"))/60,
        'lno': sum(np.load("computation_time/oscillatory/lno/unsupervised/time_epoch_1000.npy"))/60,
    },
    'polynomial': {
        'deeponet': sum(np.load("computation_time/polynomial_tracking/deeponet/unsupervised/time_epoch_1300.npy"))/60,
        'fno': sum(np.load("computation_time/polynomial_tracking/fno/unsupervised/time_epoch_780.npy"))/60,
        'lno': sum(np.load("computation_time/polynomial_tracking/lno/unsupervised/time_epoch_1000.npy"))/60,
    },
    'singular_arc': {
        'deeponet': sum(np.load("computation_time/singular_arc/deeponet/unsupervised/time_epoch_1500.npy"))/60,
        'fno': sum(np.load("computation_time/singular_arc/fno/unsupervised/time_epoch_230.npy"))/60,
        'lno': sum(np.load("computation_time/singular_arc/lno/unsupervised/time_epoch_1000.npy"))/60,
    }
}
print("linear,", full_time['linear'])
print("nonlinear,", full_time['nonlinear'])
print("oscillatory,", full_time['oscillatory'])
print("polynomial,", full_time['polynomial'])
print("singular_arc,", full_time['singular_arc'])