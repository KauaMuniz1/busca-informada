"""
Atividade Busca Informada - Quebra-cabeça de 8 peças
Greedy Best-First Search e A* com as heurísticas h1 (peças fora do lugar)
e h2 (distância de Manhattan). O espaço vazio (0) não é contado.
"""
import heapq
import itertools
import time

OBJETIVO = (1, 2, 3,
            4, 5, 6,
            7, 8, 0)

# Posição (linha, coluna) de cada peça no objetivo
POS_OBJ = {v: divmod(i, 3) for i, v in enumerate(OBJETIVO)}

# Ações: nome -> deslocamento (dl, dc) da posição vazia
ACOES = {
    "CIMA": (-1, 0),
    "BAIXO": (1, 0),
    "ESQUERDA": (0, -1),
    "DIREITA": (0, 1),
}


# ----------------------------- Heurísticas -----------------------------
def h1(estado):
    """Número de peças fora do lugar (vazio não conta)."""
    return sum(1 for i, v in enumerate(estado) if v != 0 and v != OBJETIVO[i])


def h2(estado):
    """Distância de Manhattan (vazio não conta)."""
    total = 0
    for i, v in enumerate(estado):
        if v == 0:
            continue
        x, y = divmod(i, 3)
        xo, yo = POS_OBJ[v]
        total += abs(x - xo) + abs(y - yo)
    return total


# ----------------------------- Sucessores ------------------------------
def sucessores(estado):
    """Gera (ação, novo_estado) para cada movimento válido; custo unitário."""
    i = estado.index(0)
    l, c = divmod(i, 3)
    for nome, (dl, dc) in ACOES.items():
        nl, nc = l + dl, c + dc
        if 0 <= nl < 3 and 0 <= nc < 3:
            j = nl * 3 + nc
            lst = list(estado)
            lst[i], lst[j] = lst[j], lst[i]
            yield nome, tuple(lst)


def solucionavel(estado):
    """Paridade das inversões (tabuleiro 3x3: solucionável se par)."""
    seq = [v for v in estado if v != 0]
    inv = sum(1 for a in range(len(seq)) for b in range(a + 1, len(seq))
              if seq[a] > seq[b])
    return inv % 2 == 0


# ------------------------------ Busca ----------------------------------
def buscar(inicial, h, algoritmo):
    """
    algoritmo: "Greedy" (f = h) ou "A*" (f = g + h).
    Retorna dicionário com métricas e caminho.
    """
    inicio = time.perf_counter()
    cont = itertools.count()          # desempate estável (FIFO)
    g = {inicial: 0}
    pai = {inicial: (None, None)}     # estado -> (estado_pai, ação)
    fechado = set()
    na_fronteira = {inicial}          # estados atualmente na fronteira
    f0 = h(inicial) if algoritmo == "Greedy" else h(inicial)
    heap = [(f0, h(inicial), next(cont), inicial, 0)]

    expandidos = 0
    gerados = 0
    fronteira_max = 1

    while heap:
        _, _, _, s, gs = heapq.heappop(heap)

        # entrada obsoleta (lazy deletion): estado já expandido ou g pior
        if s in fechado or gs > g[s]:
            continue

        na_fronteira.discard(s)

        if s == OBJETIVO:
            caminho, acoes = [], []
            cur = s
            while cur is not None:
                caminho.append(cur)
                p, a = pai[cur]
                if a:
                    acoes.append(a)
                cur = p
            caminho.reverse()
            acoes.reverse()
            return dict(encontrada=True, custo=g[s], profundidade=len(acoes),
                        expandidos=expandidos, gerados=gerados,
                        fronteira_max=fronteira_max,
                        tempo=time.perf_counter() - inicio,
                        caminho=caminho, acoes=acoes)

        fechado.add(s)
        expandidos += 1

        for acao, t in sucessores(s):
            gerados += 1
            novo_g = gs + 1
            if algoritmo == "Greedy":
                # Greedy não reabre estados: ignora repetidos
                if t in g:
                    continue
                g[t] = novo_g
                pai[t] = (s, acao)
                prioridade = h(t)
            else:
                # A*: só aceita se descobriu um caminho melhor (reabre se preciso)
                if t in g and novo_g >= g[t]:
                    continue
                g[t] = novo_g
                pai[t] = (s, acao)
                fechado.discard(t)
                prioridade = novo_g + h(t)
            na_fronteira.add(t)
            heapq.heappush(heap, (prioridade, h(t), next(cont), t, novo_g))
            fronteira_max = max(fronteira_max, len(na_fronteira))

    return dict(encontrada=False, custo=None, profundidade=None,
                expandidos=expandidos, gerados=gerados,
                fronteira_max=fronteira_max,
                tempo=time.perf_counter() - inicio, caminho=[], acoes=[])


# ------------------------------ Saída ----------------------------------
def fmt_estado(e):
    return "\n".join(" ".join("_" if v == 0 else str(v) for v in e[i:i + 3])
                     for i in (0, 3, 6))


def imprimir_caminho(res):
    for k, est in enumerate(res["caminho"]):
        print(f"Estado {k}")
        print(fmt_estado(est))
        print()


INSTANCIAS = {
    "A (fácil)":          (1, 5, 2,
                           4, 8, 3,
                           7, 6, 0),
    "B (intermediária)":  (5, 8, 2,
                           1, 7, 3,
                           4, 6, 0),
    "C (difícil)":        (8, 7, 2,
                           5, 4, 3,
                           1, 6, 0),
}

CONFIGS = [("Greedy", "h1", h1), ("Greedy", "h2", h2),
           ("A*", "h1", h1), ("A*", "h2", h2)]


def main(repeticoes_tempo=5):
    resultados = {}
    for nome, inicial in INSTANCIAS.items():
        assert solucionavel(inicial), f"Instância {nome} sem solução"
        print("=" * 60)
        print(f"INSTÂNCIA {nome}")
        print(fmt_estado(inicial))
        print("=" * 60)

        linhas = []
        for alg, hn, hf in CONFIGS:
            # tempo = mediana de várias execuções (a busca é determinística)
            tempos, res = [], None
            for _ in range(repeticoes_tempo):
                res = buscar(inicial, hf, alg)
                tempos.append(res["tempo"])
            res["tempo"] = sorted(tempos)[len(tempos) // 2]
            linhas.append((alg, hn, res))

            print(f"\n--- {alg} + {hn} ---")
            print(f"Solução encontrada: {'Sim' if res['encontrada'] else 'Não'}"
                  f" | Custo: {res['custo']} | Profundidade: {res['profundidade']}")
            print("Ações:", " -> ".join(res["acoes"]))
            print()
            imprimir_caminho(res)

        print(f"\nResumo - Instância {nome}")
        print(f"{'Algoritmo':<10}{'Heur.':<7}{'Custo':>6}{'Exp.':>8}"
              f"{'Ger.':>8}{'Front.Máx':>11}{'Tempo (ms)':>12}")
        for alg, hn, r in linhas:
            print(f"{alg:<10}{hn:<7}{r['custo']:>6}{r['expandidos']:>8}"
                  f"{r['gerados']:>8}{r['fronteira_max']:>11}"
                  f"{r['tempo'] * 1000:>12.3f}")
        print()
        resultados[nome] = linhas
    return resultados


if __name__ == "__main__":
    main()