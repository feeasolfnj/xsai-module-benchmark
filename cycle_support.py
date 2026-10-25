"""Describe real hardware-cycle measurements without manufacturing a fallback."""
import math
import re


def cycle_result(log, mode='auto', instrumented=True):
    result = dict(status='unavailable', reason='missing_counter_report',
                  source='perf_event_hardware_cpu_cycles', cycles=None,
                  cycles_per_unit=None, unit='module_call',
                  scope='sum_of_user_cpu_cycles_in_measured_windows')
    if mode == 'off':
        result.update(status='disabled', reason='requested_off')
        return result
    if not instrumented:
        result['reason'] = 'instrumentation_not_built'
        return result
    statuses, measurements = [], []
    for line in log.splitlines():
        if line.startswith('[PMU_STATUS]'):
            statuses.append(dict(re.findall(r'(\w+)=([^\s]+)', line)))
        elif line.startswith('[PMU_CYCLES]'):
            measurements.append(dict(re.findall(r'(\w+)=([^\s]+)', line)))
    if len(statuses) != 1:
        if statuses:
            result.update(status='invalid', reason='duplicate_counter_reports')
        return result
    status = statuses[0]
    result.update(status=status.get('status', 'invalid'), reason=status.get('reason', 'unknown'))
    if 'errno' in status:
        try:
            result['errno'] = int(status['errno'])
        except ValueError:
            result.update(status='invalid', reason='malformed_counter_report')
            return result
    if result['status'] not in ('available', 'unavailable', 'invalid', 'disabled'):
        result.update(status='invalid', reason='unknown_counter_status')
    if result['status'] != 'available':
        return result
    try:
        if len(measurements) != 1:
            raise ValueError()
        row = measurements[0]
        values = {key: int(row[key]) for key in ('cycles', 'units', 'windows', 'enabled_ns', 'running_ns')}
        if any(value <= 0 for value in values.values()):
            raise ValueError()
        if values['enabled_ns'] != values['running_ns']:
            raise ValueError()
        for key, expected in [('raw', '1'), ('scaled', '0'), ('inherited', '1'), ('exclude_kernel', '1'), ('exclude_hv', '1')]:
            if row.get(key) != expected:
                raise ValueError()
        per_unit = float(row['cycles_per_unit'])
        if not math.isfinite(per_unit) or not math.isclose(per_unit, values['cycles']/values['units'], rel_tol=1e-7, abs_tol=1e-6):
            raise ValueError()
        result.update(values, cycles_per_unit=per_unit)
        if 'empty_window_cycles' in row:
            empty = float(row['empty_window_cycles'])
            if not math.isfinite(empty) or empty < 0:
                raise ValueError()
            result['empty_window_cycles'] = empty
        result['instrumentation_overhead_subtracted'] = False
    except (KeyError, ValueError, OverflowError):
        result.update(status='invalid', reason='invalid_counter_measurement', cycles=None, cycles_per_unit=None)
    return result
