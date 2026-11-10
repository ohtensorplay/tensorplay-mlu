---
name: Bug report
about: Something does not work as expected on the MLU backend
title: ""
labels: bug
assignees: ""
---

## Environment

- MLU card / product name (`cnmon` output, first lines)
- Neuware version (`cat /usr/local/neuware/version.txt`)
- TensorPlay version (`python -c "import tensorplay; print(tensorplay.__version__)"`)
- `tensorplay-mlu` version or commit
- OS / Python version

## Reproducer

```python
import tensorplay as tp
import tensorplay_mlu
tensorplay_mlu.load()
# ... minimal code ...
```

## Expected behavior

## Actual behavior

## Notes

Is the device name generic (`mlu` in `cnmon`)?  If so, read the "Generic
product names" section of the README first — unknown-device kernels return
wrong data or fault without an error status.