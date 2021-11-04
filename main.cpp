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
template <typename T> vector<int> IOTA(vector<T> a) {
    int n = a.size();
    vector<int> id(n);
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
//#define endl '\n'
const long double eps = 0.000000000000001;
const long double PI = 3.141592653589793;
vl dx = {1, 0, -1, 0};
vl dy = {0, 1, 0, -1};
using mint = modint<mod>;
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
    ll id, dsm, indeg, outdeg, priority;
    Task(ll id, ll dsm, ll indeg, ll outdeg) : id(id), dsm(dsm), indeg(indeg), outdeg(outdeg) {
        priority = outdeg;
    }
};

bool operator<(const Task &t1, const Task &t2) {
    return t1.priority < t2.priority;
};
bool operator>(const Task &t1, const Task &t2) {
    return t2 < t1;
}
bool operator<=(const Task &t1, const Task &t2) {
    return !(t1 > t2);
}
bool operator>=(const Task &t1, const Task &t2) {
    return !(t1 < t2);
}

int main() {
    cin.tie(0);
    ios::sync_with_stdio(0);
    cout << setprecision(30) << fixed;

    // 初期入力
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

    // タスク管理
    vl dsum(n);
    rep(i, n) {
        dsum[i] = vsum(d[i]);
    }
    vl in_deg(n, 0), out_deg(n, 0);
    rep(i, n) {
        for (auto e : g.g[i]) {
            in_deg[e.to]++;
            out_deg[e.from]++;
        }
    }
    auto construct_task = [&](int i) { return Task(i, dsum[i], in_deg[i], out_deg[i]); };
    priority_queue<Task> can_begin;
    rep(i, n) {
        if (in_deg[i] == 0)
            can_begin.emplace(construct_task(i));
    }

    // メンバー管理
    vl can_work(m);
    iota(all(can_work), 0);

    // utils
    vl member_to_task(m);

    while (true) {
        // 出力
        ll task_num = can_begin.size();
        ll member_num = can_work.size();
        vl tasks;
        vl members;
        rep(i, min(task_num, member_num)) {
            ll task = can_begin.top().id;
            tasks.pb(task);
            can_begin.pop();
            ll mem = can_work.back();
            members.pb(mem);
            can_work.pop_back();
            member_to_task[mem] = task;
        }
        output(members, tasks);

        // 入力
        ll end_num;
        cin >> end_num;
        if (end_num == -1) {
            return 0;
        }
        vl f(end_num);
        scan(f);
        for (ll ff : f) {
            ff--;
            can_work.pb(ff);
            for (auto &e : g.g[member_to_task[ff]]) {
                in_deg[e.to]--;
                if (in_deg[e.to] == 0)
                    can_begin.emplace(construct_task(e.to));
            }
        }
    }
}

/* memo
out_degでソートしてみた
*/
