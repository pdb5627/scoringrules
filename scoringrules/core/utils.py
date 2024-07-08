from scoringrules.backend import backends

_V_AXIS = -1
_M_AXIS = -2


def _multivariate_shape_compatibility(obs, fct, m_axis) -> None:
    f_shape = fct.shape
    o_shape = obs.shape
    o_shape_broadcast = o_shape[:m_axis] + (f_shape[m_axis],) + o_shape[m_axis:]
    if o_shape_broadcast != f_shape:
        raise ValueError(
            f"Forecasts shape {f_shape} and observations shape {o_shape} are not compatible for broadcasting!"
        )


def _forecast_weights_shape_compatibility(fct, fct_wt, v_axis) -> None:
    f_shape = fct.shape
    w_shape = fct_wt.shape
    f_shape_compat = list(f_shape)
    del f_shape_compat[v_axis]
    if tuple(f_shape_compat) != w_shape:
        raise ValueError(
            f"Forecasts shape {f_shape} and forecast weights shape {w_shape} do not match in non-variable axis"
        )


def _multivariate_shape_permute(obs, fct, m_axis, v_axis, backend=None, fct_wt=None):
    B = backends.active if backend is None else backends[backend]
    v_axis_obs = v_axis - 1 if m_axis < v_axis else v_axis
    fct = B.moveaxis(fct, (m_axis, v_axis), (_M_AXIS, _V_AXIS))
    obs = B.moveaxis(obs, v_axis_obs, _V_AXIS)
    if fct_wt is not None:
        # Temporarily add variable axis
        fct_wt = B.expand_dims(fct_wt, v_axis)
        # Move the axes to standard positions
        fct_wt = B.moveaxis(fct_wt, (m_axis, v_axis), (_M_AXIS, _V_AXIS))
        # Squeeze out variable axis
        fct_wt = B.squeeze(fct_wt, axis=_V_AXIS)
    return obs, fct


def multivariate_array_check(obs, fct, m_axis, v_axis, backend=None, fct_wt=None):
    """Check and adapt the shapes of multivariate forecasts and observations arrays."""
    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))
    if fct_wt is not None:
        fct_wt = B.asarray(fct_wt)
    m_axis = m_axis if m_axis >= 0 else fct.ndim + m_axis
    v_axis = v_axis if v_axis >= 0 else fct.ndim + v_axis
    _multivariate_shape_compatibility(obs, fct, m_axis)
    if fct_wt is not None:
        _forecast_weights_shape_compatibility(fct, fct_wt, v_axis)
    return _multivariate_shape_permute(
        obs, fct, m_axis, v_axis, backend=backend, fct_wt=fct_wt
    )
