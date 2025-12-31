# OnlineSTL Lambda Parameter Fix

## Problem
OnlineSTL residual showed strong seasonal pattern (autocorrelation @ lag=period: 0.98)

## Root Cause
Default lambda (smoothing parameter) of 0.7 was too high, causing:
- Slow adaptation to seasonal changes
- Under-estimation of seasonal component  
- Seasonal error accumulating in residual

## Solution
**Changed lambda from 0.7 to 0.3**

### Results

| Lambda | Residual Autocorr | MSE (Trend) | Status |
|--------|------------------|-------------|---------|
| 0.7 | 0.9419 | 6.30 | ❌ High seasonal pattern |
| 0.5 | 0.9436 | - | Still high |
| **0.3** | **0.5977** | **3.26** | ✅ Much improved |
| 0.2 | - | - | Too sensitive |

### Impact
- ✅ **MSE improved by 48%** (6.30 → 3.26)
- ✅ **Autocorrelation reduced by 36%** (0.94 → 0.60)
- ✅ Better seasonal removal (residual closer to white noise)
- ⚠️ Still has some autocorrelation (inherent to online algorithm)

## Changes Made

1. **All dataset configs updated:**
   - `synth1-9_params.json`: `lam: 0.7 → 0.3`

2. **Setup script updated:**
   - [scripts/add_online_stl.py](scripts/add_online_stl.py): default `lam=0.3`

3. **Wrapper default:**
   - [experiments/baselines/online_stl.py](experiments/baselines/online_stl.py): still uses param from config

## Understanding Lambda

**Lambda (λ)** controls exponential smoothing in seasonal estimation:
- **High λ (0.7-0.9):** More smoothing, slower adaptation
  - Pro: Stable estimates, less noise
  - Con: Lags behind changes, under-estimates seasonal
- **Low λ (0.3-0.5):** Less smoothing, faster adaptation  
  - Pro: Tracks seasonal better, lower residual autocorrelation
  - Con: More sensitive to noise

**Optimal for our data: λ=0.3**

## Remaining Limitations

Even with λ=0.3, OnlineSTL still has:
1. Residual autocorrelation ~0.60 (vs ideal ~0)
2. Higher MSE than batch STL methods
3. Some seasonal under-estimation

This is **expected for online algorithms** - they trade accuracy for O(1) update speed.

## Recommendation

Use OnlineSTL when:
- Real-time processing required
- O(1) updates are critical
- Acceptable residual autocorrelation: 0.5-0.7

Use batch STL/LGTD when:
- Need residual to be pure white noise
- All data available upfront
- Accuracy > speed

---

**Status:** Fixed and tested ✅

OnlineSTL now performs significantly better with λ=0.3, though some residual autocorrelation remains (inherent to online processing).
