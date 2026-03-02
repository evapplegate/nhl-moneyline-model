# Contributing to NHL Moneyline Model

## Overview

This project welcomes contributions and improvements. Whether you're fixing bugs, adding features, or improving documentation, we appreciate your help.

## How to Contribute

### Setup
1. Clone the repository and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Extending the Project

**Common enhancements:**
- **Add features:** Extend `src/phase3_make_features.py` with new predictive signals (e.g., player injury reports, goal-scoring trends)
- **Tune models:** Run `src/phase4_train_model.py` with new hyperparameters to improve AUC/log loss
- **Add models:** Implement additional algorithms (Neural Networks, LightGBM) and compare via `src/eval_models.py`
- **Expand API:** Add endpoints in `app/main.py` for team matchup analysis, feature importance explanations
- **Improve deployment:** Enhance Docker setup, add Kubernetes manifests, integrate with cloud platforms

### Testing & Quality
- Run test suite before submitting PRs:
  ```bash
  pytest
  ```
- Update tests when adding new features
- Ensure code follows PEP 8 style

### Submission
1. Commit with clear messages: `git commit -m "Add feature: brief description"`
2. Push to your branch and open a Pull Request
3. Describe changes, reference any issues, and provide testing evidence

---

**Questions?** Check [MODEL_CARD.md](MODEL_CARD.md) for technical deep dive or [DEMO.md](DEMO.md) for quick start.

