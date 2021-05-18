## Downloading your dataset

Example invocation to download emails from IMAP:

```
$ python -m recruiterspam.download --host imap.gmail.com --port 993 --ssl --username 'me@waleedkhan.name' --password "$APP_PASSWORD" --output messages.json
```

Example invocation to unpack emails from a `.mbox` file:

```
$ python -m recruiterspam.unpackmbox --mbox my.mbox --folder all --label recruiter-spam --output messages.json
```

## Training the model

Run the following:

```
$ python -m recruiterspam.train --messages messages --output model.pkl
```

The model is then stored in the Python pickle file `model.pkl`.

## Examining the model

You can then run a backtest to manually examine your flagged messages:

```
$ python -m recruiterspam.backtest --messages messages --model model.pkl
...
INFO:root:(false positive) uid=3974 Re: [openstenoproject/plover] Dictionary Suggestions (#400)
INFO:root:(true positive)  uid=3980 Scala jobs in Seattle!
INFO:root:(false positive) uid=4082 Re: [openstenoproject/plover] Dictionary Suggestions (#400)
INFO:root:(false positive) uid=4215 Waleed, please add me to your LinkedIn network
INFO:root:(true positive)  uid=5091 Join a Startup to Bet Your Career On!
INFO:root:(true positive)  uid=5504 Hi Waleed
INFO:root:(true positive)  uid=5524 Hiring IOT talent at Ennate
...
```

Or you can check a custom message manually:

```
$ python -m recruiterspam.classify --model model.pkl
INFO:root:Reading text from stdin...
It’s a small team of talented engineers who enjoy utilizing the latest technology in the distributed computing space to build bespoke solutions for an ever growing, fully systematic trading platform.

INFO:root:Top keywords:
INFO:root:  bespoke: 10.502283
INFO:root:  ever: 10.096818
INFO:root:  systematic: 9.809136
INFO:root:  utilizing: 9.809136
INFO:root:  fully: 9.585992
INFO:root:  computing: 9.403671
INFO:root:  enjoy: 9.403671
INFO:root:  latest: 9.403671
INFO:root:  small: 8.710523
INFO:root:  its: 8.556373
INFO:root:Prediction 1 with probability 0.999999
```
