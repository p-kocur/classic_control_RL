# Testowanie Algorytmów DQN, A2C i PPO w środowisku CartPole

To repozytorium zawiera implementacje oraz testy działania algorytmów Deep Q-Learning (DQN), Advantage Actor-Critic (A2C) oraz Proximal Policy Optimization (PPO) w środowisku `CartPole` z biblioteki OpenAI Gym.

## Wymagania

Aby uruchomić ten projekt, upewnij się, że masz zainstalowanego Pythona w wersji 3.7 lub nowszej oraz zainstalowane wymagane pakiety. Możesz to zrobić, wykonując poniższe kroki:

1. Skopiuj repozytorium na swój komputer:
   ```bash
   git clone https://github.com/twoje-repozytorium.git
   cd twoje-repozytorium
   ```

2. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt
   ```

## Sposób użycia

### Testowanie algorytmów

Aby uruchomić testy dla algorytmów DQN, A2C oraz PPO, wykonaj:
```bash
python main.py
```

Skrypt `main.py` przeprowadzi testy działania algorytmów, ucząc modele w środowisku `CartPole` i zapisując ich wyniki.

### Obserwacja działania wytrenowanych sieci

Jeśli chcesz zobaczyć, jak działają wytrenowane sieci dla wybranego algorytmu, użyj skryptu `see_results.py`. Aby uruchomić wizualizację, wykonaj:
```bash
python see_results.py -o [a2c, ppo, dqn]
```
Zastąp `[a2c, ppo, dqn]` nazwą algorytmu, który chcesz przetestować, np.:
```bash
python see_results.py -o ppo
```

## Struktura projektu

- `main.py` - Skrypt główny do uruchamiania testów i trenowania modeli.
- `see_results.py` - Skrypt umożliwiający wizualizację działania wytrenowanych sieci.
- `requirements.txt` - Lista wymaganych bibliotek.
- `models/` - Folder zawierający zapisane modele dla każdego algorytmu.
- `algorithms/` - Folder zawierający pliki z algorytmami do trenowania sieci.
- `figures/` - Folder, w którym znajduje się wykres z wynikami ostatniego eksperymentu.

## Autor
Paweł Kocur
