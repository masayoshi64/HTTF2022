/* #region header */

#pragma GCC optimize("Ofast")
#include <bits/stdc++.h>
using namespace std;
// types
using ll = long long;
using ull = unsigned long long;
using ld = long double;
typedef pair<ll, ll> Pl;
typedef pair<int, int> Pi;
typedef vector<ll> vl;
typedef vector<int> vi;
typedef vector<char> vc;
template <typename T> using mat = vector<vector<T>>;
typedef vector<vector<int>> vvi;
typedef vector<vector<long long>> vvl;
typedef vector<vector<char>> vvc;
// abreviations
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define rep_(i, a_, b_, a, b, ...) for (ll i = (a), max_i = (b); i < max_i; i++)
#define rep(i, ...) rep_(i, __VA_ARGS__, __VA_ARGS__, 0, __VA_ARGS__)
#define rrep_(i, a_, b_, a, b, ...) for (ll i = (b - 1), min_i = (a); i >= min_i; i--)
#define rrep(i, ...) rrep_(i, __VA_ARGS__, __VA_ARGS__, 0, __VA_ARGS__)
#define srep(i, a, b, c) for (ll i = (a), max_i = (b); i < max_i; i += c)
#define SZ(x) ((int)(x).size())
#define pb(x) push_back(x)
#define eb(x) emplace_back(x)
#define mp make_pair
//入出力
#define print(x) cout << x << endl
template <class T> ostream &operator<<(ostream &os, const vector<T> &v) {
    for (auto &e : v)
        cout << e << " ";
    cout << endl;
    return os;
}
void scan(int &a) {
    cin >> a;
}
void scan(long long &a) {
    cin >> a;
}
void scan(char &a) {
    cin >> a;
}
void scan(double &a) {
    cin >> a;
}
void scan(string &a) {
    cin >> a;
}
template <class T> void scan(vector<T> &a) {
    for (auto &i : a)
        scan(i);
}
#define vsum(x) accumulate(all(x), 0LL)
#define vmax(a) *max_element(all(a))
#define vmin(a) *min_element(all(a))
#define lb(c, x) distance((c).begin(), lower_bound(all(c), (x)))
#define ub(c, x) distance((c).begin(), upper_bound(all(c), (x)))
// functions
// gcd(0, x) fails.
ll gcd(ll a, ll b) {
    return b ? gcd(b, a % b) : a;
}
ll lcm(ll a, ll b) {
    return a / gcd(a, b) * b;
}
ll safemod(ll a, ll b) {
    return (a % b + b) % b;
}
template <class T> bool chmax(T &a, const T &b) {
    if (a < b) {
        a = b;
        return 1;
    }
    return 0;
}
template <class T> bool chmin(T &a, const T &b) {
    if (b <= a) {
        a = b;
        return 1;
    }
    return 0;
}
template <typename T> T mypow(T x, ll n) {
    T ret = 1;
    while (n > 0) {
        if (n & 1)
            (ret *= x);
        (x *= x);
        n >>= 1;
    }
    return ret;
}
ll modpow(ll x, ll n, const ll mod) {
    ll ret = 1;
    while (n > 0) {
        if (n & 1)
            (ret *= x);
        (x *= x);
        n >>= 1;
        x %= mod;
        ret %= mod;
    }
    return ret;
}

uint64_t my_rand(void) {
    static uint64_t x = 88172645463325252ULL;
    x = x ^ (x << 13);
    x = x ^ (x >> 7);
    return x = x ^ (x << 17);
}
int popcnt(ull x) {
    return __builtin_popcountll(x);
}
template <typename T> vector<ll> IOTA(vector<T> a) {
    int n = a.size();
    vector<ll> id(n);
    iota(all(id), 0);
    sort(all(id), [&](int i, int j) { return a[i] < a[j]; });
    return id;
}
struct Timer {
    clock_t start_time;
    void start() {
        start_time = clock();
    }
    int lap() {
        // return x ms.
        return (clock() - start_time) * 1000 / CLOCKS_PER_SEC;
    }
};
template <typename T = int> struct Edge {
    int from, to;
    T cost;
    int idx;

    Edge() = default;

    Edge(int from, int to, T cost = 1, int idx = -1) : from(from), to(to), cost(cost), idx(idx) {
    }

    operator int() const {
        return to;
    }
};

template <typename T = int> struct Graph {
    vector<vector<Edge<T>>> g;
    int es;

    Graph() = default;

    explicit Graph(int n) : g(n), es(0) {
    }

    size_t size() const {
        return g.size();
    }

    void add_directed_edge(int from, int to, T cost = 1) {
        g[from].emplace_back(from, to, cost, es++);
    }

    void add_edge(int from, int to, T cost = 1) {
        g[from].emplace_back(from, to, cost, es);
        g[to].emplace_back(to, from, cost, es++);
    }

    void read(int M, int padding = -1, bool weighted = false, bool directed = false) {
        for (int i = 0; i < M; i++) {
            int a, b;
            cin >> a >> b;
            a += padding;
            b += padding;
            T c = T(1);
            if (weighted)
                cin >> c;
            if (directed)
                add_directed_edge(a, b, c);
            else
                add_edge(a, b, c);
        }
    }
};

template <int Mod> struct modint {
    int x;

    modint() : x(0) {
    }

    modint(long long y) : x(y >= 0 ? y % Mod : (Mod - (-y) % Mod) % Mod) {
    }

    modint &operator+=(const modint &p) {
        if ((x += p.x) >= Mod)
            x -= Mod;
        return *this;
    }

    modint &operator-=(const modint &p) {
        if ((x += Mod - p.x) >= Mod)
            x -= Mod;
        return *this;
    }

    modint &operator*=(const modint &p) {
        x = (int)(1LL * x * p.x % Mod);
        return *this;
    }

    modint &operator/=(const modint &p) {
        *this *= p.inverse();
        return *this;
    }

    modint operator-() const {
        return modint(-x);
    }

    modint operator+(const modint &p) const {
        return modint(*this) += p;
    }

    modint operator-(const modint &p) const {
        return modint(*this) -= p;
    }

    modint operator*(const modint &p) const {
        return modint(*this) *= p;
    }

    modint operator/(const modint &p) const {
        return modint(*this) /= p;
    }

    bool operator==(const modint &p) const {
        return x == p.x;
    }

    bool operator!=(const modint &p) const {
        return x != p.x;
    }

    modint inverse() const {
        int a = x, b = Mod, u = 1, v = 0, t;
        while (b > 0) {
            t = a / b;
            swap(a -= t * b, b);
            swap(u -= t * v, v);
        }
        return modint(u);
    }

    modint pow(int64_t n) const {
        modint ret(1), mul(x);
        while (n > 0) {
            if (n & 1)
                ret *= mul;
            mul *= mul;
            n >>= 1;
        }
        return ret;
    }

    friend ostream &operator<<(ostream &os, const modint &p) {
        return os << p.x;
    }

    friend istream &operator>>(istream &is, modint &a) {
        long long t;
        is >> t;
        a = modint<Mod>(t);
        return (is);
    }

    static int get_mod() {
        return Mod;
    }

    constexpr int get() const {
        return x;
    }
};

/* #endregion*/
#define inf 1000000000ll
#define INF 4000000004000000000LL
#define mod 998244353ll

// main
random_device rnd; // 非決定的な乱数生成器でシード生成機を生成
mt19937 mt(rnd());
normal_distribution<> ndist(0.0, 1.0);
uniform_real_distribution<> udist_s(20, 60), udist_d(10, 40);

void output(vl &a, vl &b) {
    assert(a.size() == b.size());
    ll m = a.size();
    cout << m;
    rep(i, m) {
        cout << ' ' << a[i] + 1 << ' ' << b[i] + 1;
    }
    cout << endl;
}

struct Task {
    ll id;
    ld priority;
    Task(ll id, ld priority) : id(id), priority(priority) {
    }
};

bool operator<(const Task &t1, const Task &t2) {
    return t1.priority < t2.priority;
};

struct Member {
    ll id;
    ld priority;
    Member(ll id, ld p) : id(id), priority(p) {
    }
};

bool operator<(const Member &t1, const Member &t2) {
    return t1.priority < t2.priority;
};

vl generate_s(int k) {
    vector<double> s(k);
    double sm = 0;
    rep(i, k) {
        s[i] = abs(ndist(mt));
        sm += s[i] * s[i];
    }
    vl res(k);
    double coef = udist_s(mt) / sqrt(sm);
    rep(i, k) {
        res[i] = round(coef * s[i]);
    }
    return res;
}

vl generate_d(int k) {
    vector<double> d(k);
    double sm = 0;
    rep(i, k) {
        d[i] = abs(ndist(mt));
        sm += d[i] * d[i];
    }
    vl res(k);
    double coef = udist_d(mt) / sqrt(sm);
    rep(i, k) {
        res[i] = round(coef * d[i]);
    }
    return res;
}

ll calc_required_days(vl s, vl d) {
    ll w = 0;
    ll k = s.size();
    rep(i, k) {
        w += max(0ll, d[i] - s[i]);
    }
    return max(1ll, w);
}
template <class Cap, class Cost> struct mcf_graph {
  public:
    mcf_graph() {
    }
    mcf_graph(int n) : _n(n), g(n) {
    }

    int add_edge(int from, int to, Cap cap, Cost cost) {
        assert(0 <= from && from < _n);
        assert(0 <= to && to < _n);
        int m = int(pos.size());
        pos.push_back({from, int(g[from].size())});
        g[from].push_back(_edge{to, int(g[to].size()), cap, cost});
        g[to].push_back(_edge{from, int(g[from].size()) - 1, 0, -cost});
        return m;
    }

    struct edge {
        int from, to;
        Cap cap, flow;
        Cost cost;
    };

    edge get_edge(int i) {
        int m = int(pos.size());
        assert(0 <= i && i < m);
        auto _e = g[pos[i].first][pos[i].second];
        auto _re = g[_e.to][_e.rev];
        return edge{
            pos[i].first, _e.to, _e.cap + _re.cap, _re.cap, _e.cost,
        };
    }
    std::vector<edge> edges() {
        int m = int(pos.size());
        std::vector<edge> result(m);
        for (int i = 0; i < m; i++) {
            result[i] = get_edge(i);
        }
        return result;
    }

    std::pair<Cap, Cost> flow(int s, int t) {
        return flow(s, t, std::numeric_limits<Cap>::max());
    }
    std::pair<Cap, Cost> flow(int s, int t, Cap flow_limit) {
        return slope(s, t, flow_limit).back();
    }
    std::vector<std::pair<Cap, Cost>> slope(int s, int t) {
        return slope(s, t, std::numeric_limits<Cap>::max());
    }
    std::vector<std::pair<Cap, Cost>> slope(int s, int t, Cap flow_limit) {
        assert(0 <= s && s < _n);
        assert(0 <= t && t < _n);
        assert(s != t);
        // variants (C = maxcost):
        // -(n-1)C <= dual[s] <= dual[i] <= dual[t] = 0
        // reduced cost (= e.cost + dual[e.from] - dual[e.to]) >= 0 for all edge
        std::vector<Cost> dual(_n, 0), dist(_n);
        std::vector<int> pv(_n), pe(_n);
        std::vector<bool> vis(_n);
        auto dual_ref = [&]() {
            std::fill(dist.begin(), dist.end(), std::numeric_limits<Cost>::max());
            std::fill(pv.begin(), pv.end(), -1);
            std::fill(pe.begin(), pe.end(), -1);
            std::fill(vis.begin(), vis.end(), false);
            struct Q {
                Cost key;
                int to;
                bool operator<(Q r) const {
                    return key > r.key;
                }
            };
            std::priority_queue<Q> que;
            dist[s] = 0;
            que.push(Q{0, s});
            while (!que.empty()) {
                int v = que.top().to;
                que.pop();
                if (vis[v])
                    continue;
                vis[v] = true;
                if (v == t)
                    break;
                // dist[v] = shortest(s, v) + dual[s] - dual[v]
                // dist[v] >= 0 (all reduced cost are positive)
                // dist[v] <= (n-1)C
                for (int i = 0; i < int(g[v].size()); i++) {
                    auto e = g[v][i];
                    if (vis[e.to] || !e.cap)
                        continue;
                    // |-dual[e.to] + dual[v]| <= (n-1)C
                    // cost <= C - -(n-1)C + 0 = nC
                    Cost cost = e.cost - dual[e.to] + dual[v];
                    if (dist[e.to] - dist[v] > cost) {
                        dist[e.to] = dist[v] + cost;
                        pv[e.to] = v;
                        pe[e.to] = i;
                        que.push(Q{dist[e.to], e.to});
                    }
                }
            }
            if (!vis[t]) {
                return false;
            }

            for (int v = 0; v < _n; v++) {
                if (!vis[v])
                    continue;
                // dual[v] = dual[v] - dist[t] + dist[v]
                //         = dual[v] - (shortest(s, t) + dual[s] - dual[t]) + (shortest(s, v) + dual[s] - dual[v])
                //         = - shortest(s, t) + dual[t] + shortest(s, v)
                //         = shortest(s, v) - shortest(s, t) >= 0 - (n-1)C
                dual[v] -= dist[t] - dist[v];
            }
            return true;
        };
        Cap flow = 0;
        Cost cost = 0, prev_cost = -1;
        std::vector<std::pair<Cap, Cost>> result;
        result.push_back({flow, cost});
        while (flow < flow_limit) {
            if (!dual_ref())
                break;
            Cap c = flow_limit - flow;
            for (int v = t; v != s; v = pv[v]) {
                c = std::min(c, g[pv[v]][pe[v]].cap);
            }
            for (int v = t; v != s; v = pv[v]) {
                auto &e = g[pv[v]][pe[v]];
                e.cap -= c;
                g[v][e.rev].cap += c;
            }
            Cost d = -dual[s];
            flow += c;
            cost += c * d;
            if (prev_cost == d) {
                result.pop_back();
            }
            result.push_back({flow, cost});
            prev_cost = cost;
        }
        return result;
    }

  private:
    int _n;

    struct _edge {
        int to, rev;
        Cap cap;
        Cost cost;
    };

    std::vector<std::pair<int, int>> pos;
    std::vector<std::vector<_edge>> g;
};
struct matching {
    int n, m;
    mcf_graph<int, int> mcf;
    matching(int n, int m) : n(n), m(m), mcf(n + m + 2) {
        rep(i, n) {
            mcf.add_edge(n + m, i, 1, 0);
        }
        rep(i, m) {
            mcf.add_edge(i + n, n + m + 1, 1, 0);
        }
    }
    void add_edge(int x, int y, int cost) {
        mcf.add_edge(x, y + n, 1, cost);
    }
    vi match() {
        mcf.flow(n + m, n + m + 1, min(n, m));
        auto edges = mcf.edges();
        vi task(n, -1);
        for (auto edge : edges) {
            if (edge.from == n + m || edge.to == n + m + 1 || edge.flow == 0)
                continue;
            task[edge.from] = edge.to - n;
        }
        return task;
    }
};
int main() {
    cin.tie(0);
    ios::sync_with_stdio(0);
    cout << setprecision(30) << fixed;
    // 入力
    ll n, m, k, r;
    cin >> n >> m >> k >> r;

    mat<ll> d(n, vl(k));
    scan(d);
    Graph<ll> g(n);
    rep(i, r) {
        ll u, v;
        cin >> u >> v;
        u--;
        v--;
        g.add_directed_edge(u, v);
    }

    // 前処理
    vl in_deg(n, 0); //, out_deg(n, 0);
    rep(i, n) {
        for (auto e : g.g[i]) {
            in_deg[e.to]++;
            // out_deg[e.from]++;
        }
    }

    //// タスクの処理

    // タスクのpriority
    vl priority(n);
    auto dfs = [&](int v, auto &dfs) -> ll {
        if (priority[v] > 0)
            return priority[v];
        ll res = 1;
        for (auto e : g.g[v]) {
            res += dfs(e.to, dfs);
        }
        priority[v] = res;
        return res;
    };
    rep(i, n) dfs(i, dfs);

    // 使用できるタスク
    priority_queue<Task> can_begin;
    rep(i, n) {
        if (in_deg[i] == 0) {
            can_begin.emplace(i, priority[i]);
        }
    }

    //// メンバーの処理
    // priority_queue<Member> can_work;
    vl can_work_list;
    rep(i, m) {
        // can_work.emplace(i, 0);
        can_work_list.pb(i);
    }
    vl member_to_task(m);
    vl member_to_day(m);

    // sの候補を生成
    int num_cand = 1000;
    vl member_to_cand(m);
    mat<ll> similarity(m, vl(num_cand));
    mat<ll> cand_s(num_cand);
    rep(i, num_cand) {
        cand_s[i] = generate_s(k);
    }

    // メンバーが何回使われたか
    vi task_cnt(m);

    // ループ
    ll day = 0;
    while (true) {
        day++;
        vl tasks;
        vl members;
        // 出力
        while (!can_begin.empty() && !can_work_list.empty()) {
            int task_num = can_begin.size();
            int member_num = can_work_list.size();
            // 今日割り振るタスク、メンバーをpriorityに従い選定
            int task_top_num = (1);
            matching match(task_top_num, member_num);
            vector<Task> que;
            rep(i, task_top_num) {
                Task task = can_begin.top();
                que.push_back(task);
                can_begin.pop();
                rep(j, member_num) {
                    int mid = can_work_list[j];
                    match.add_edge(i, j, calc_required_days(cand_s[member_to_cand[mid]], d[task.id]));
                }
            }
            vi t2m = match.match();
            rep(i, task_top_num) {
                if (t2m[i] == -1)
                    can_begin.push(que[i]);
                else {
                    tasks.pb(que[i].id);
                    int tid = que[i].id;
                    int mid = can_work_list[t2m[i]];
                    members.pb(mid);
                    member_to_task[mid] = tid;
                    member_to_day[mid] = day;
                }
            }
            for (int mid : members) {
                can_work_list.erase(remove(all(can_work_list), mid), can_work_list.end());
            }
        }

        output(members, tasks);

        // 入力
        ll end_num;
        cin >> end_num;

        vl f(end_num);
        scan(f);
        for (ll mid : f) {
            mid--;
            ll tid = member_to_task[mid];
            // can_work.emplace(mid, member_to_day[mid] - vsum(d[tid]));
            can_work_list.pb(mid);

            // candidateに重みづけ
            ll kikan = day - member_to_day[mid];
            ll mx = -INF;
            rep(sid, num_cand) {
                similarity[mid][sid] -= mypow<ll>((kikan - calc_required_days(cand_s[sid], d[tid])), 2);
                if (chmax(mx, similarity[mid][sid]))
                    member_to_cand[mid] = sid;
            }

            for (auto &e : g.g[tid]) {
                in_deg[e.to]--;
                if (in_deg[e.to] == 0) {
                    can_begin.emplace(e.to, priority[e.to]);
                }
            }
        }

        // output estimated s
        // cout << "#s " << mid + 1 << " ";
        // rep(i, k) {
        //     cout << cand_s[member_to_cand[mid]][i] << ' ';
        // }
        // cout << endl;
    }
}

/* memo
out_degでソートしてみた 75529
メンバーをランダムに選出 77562
*/
