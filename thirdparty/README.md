# Third-party Dependencies

Place the Pi3 training branch here when you want to run against the real Pi3
dataset stack:

```powershell
git clone https://github.com/yyfz/Pi3.git thirdparty/Pi3
cd thirdparty/Pi3
git checkout training
python -m pip install -e .
```

`unidata_skill.pi3x` resolves Pi3 only from `thirdparty/Pi3`.

The `thirdparty/Pi3/` checkout is ignored by git.

Current dataloaders require a Pi3 training checkout at runtime. Tests use a
small fake Pi3 package under `tests/fake_pi3`; production commands should use
the real repository.
