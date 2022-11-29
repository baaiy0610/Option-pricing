### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 6c63ca90-4b3f-41b0-b6a2-0bbaec4340d0
import Pkg;

# ╔═╡ 4e6d60fa-6ac0-43fc-a105-996cfe9ce573
Pkg.add("Distributions")

# ╔═╡ 66c0ff66-8fb9-11eb-0bcd-fbbb78c71042
begin
	using Statistics
	using Distributions
end

# ╔═╡ c3093dea-8fb9-11eb-24be-857a7c9aa459
md"""# exact solution


"""

# ╔═╡ b5f156a8-8fb9-11eb-05aa-4daa727664ad
function BS_exat(sigma, r, K, S0, T)
	d1 = (log(S0/K)+(r+0.5*sigma^2)*T)/(sigma*sqrt(T));
    d2 = d1 - sigma*sqrt(T);
    V = S0 * cdf.(Normal(),d1) - K * exp(-r*T) * cdf.(Normal(),d2);
    return V
end

# ╔═╡ Cell order:
# ╟─c3093dea-8fb9-11eb-24be-857a7c9aa459
# ╠═6c63ca90-4b3f-41b0-b6a2-0bbaec4340d0
# ╠═4e6d60fa-6ac0-43fc-a105-996cfe9ce573
# ╠═66c0ff66-8fb9-11eb-0bcd-fbbb78c71042
# ╠═b5f156a8-8fb9-11eb-05aa-4daa727664ad
