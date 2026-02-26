import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo

def plot_lce_histories(
    times: list | np.ndarray,
    hist_qr: np.ndarray,
    hist_eig: np.ndarray = None,
    hist_det: np.ndarray = None,
    filename: str = 'figures/lyapunov_evolution.html'
) -> None:
    traces = [
        go.Scatter(x=times, y=hist_qr[:, 0], mode='lines', name='QR λ₁'),
        go.Scatter(x=times, y=hist_qr[:, 1], mode='lines', name='QR λ₂'),
    ]
    if hist_eig is not None:
        traces += [
            go.Scatter(x=times, y=hist_eig[:, 0], mode='lines', name='Eigen λ₁'),
            go.Scatter(x=times, y=hist_eig[:, 1], mode='lines', name='Eigen λ₂'),
        ]
    if hist_det is not None:
        traces.append(go.Scatter(x=times, y=hist_det, mode='lines', name='Det Sum'))

    layout = go.Layout(
        title='Lyapunov Exponent Evolution',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Exponent'),
        template='plotly_white'
    )
    fig = go.Figure(data=traces, layout=layout)
    pyo.plot(fig, filename=filename, auto_open=True)
