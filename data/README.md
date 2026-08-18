# Data

Nothing here is tracked by git. The layout the pipeline expects:

```
data/
  raw/                  the unpacked OSF archive, study folders directly inside
  topoplots/            written by `ocular prepare`
  manifest.csv          written by `ocular prepare`
```

Download the dataset from <https://osf.io/2qgrd/> and unpack it into `raw/`.

It contains eye artifact and resting brain activity across 45 recording sessions from
39 participants, split into five studies. `ocular prepare` walks it recursively for
`.set` files, so the exact nesting does not matter as long as each study is its own
folder.
