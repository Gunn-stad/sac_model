# Where to save this in your repo

Save this file:

```text
C:\Gunnar\sac_model\make_fixed_entropy_experiments.py
```

Then run:

```powershell
cd C:\Gunnar\sac_model
python make_fixed_entropy_experiments.py
```

It will create:

```text
C:\Gunnar\sac_model\experiments\v6_fixed_entropy_40k_seed0.py
C:\Gunnar\sac_model\experiments\v6_fixed_entropy_40k_seed1.py
C:\Gunnar\sac_model\experiments\v6_fixed_entropy_40k_seed2.py
C:\Gunnar\sac_model\experiments\v6_fixed_entropy_60k_seed1.py
C:\Gunnar\sac_model\experiments\theme_fixed_entropy_40k_seed1.py
```

Then push:

```powershell
git add .
git commit -m "Add fixed entropy experiment files"
git push
```

In Colab:

```python
%cd /content
!rm -rf sac_model
!git clone https://github.com/Gunn-stad/sac_model.git
%cd /content/sac_model

!pip install stable-baselines3 gymnasium shimmy yfinance pyarrow textblob pytrends vaderSentiment

!python experiments/v6_fixed_entropy_60k_seed1.py
```

Then test the theme universe:

```python
!python experiments/theme_fixed_entropy_40k_seed1.py
```
