### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 812f50f0-9c84-11eb-278d-b552fcf958e4
import Pkg

# ╔═╡ 8eb6c190-9c84-11eb-37de-7f99b0e41e5c
Pkg.add("PyPlot")

# ╔═╡ 5f8df1c8-8f4a-11eb-2aaa-a743941f3de9
begin
	using Statistics;
	using PyPlot;
	sigma=0.3;s0=12;r=0.05;T=1;N=300;t=T/N;n=Int(N+1);K=10;
end

# ╔═╡ 07abc4fa-8f34-11eb-0f33-636a1c8257b1
md"""## Monte-Carlo Simulation

Usually, the fair value of a financial derivative can be presented as $e^{-rT}E^{Q}\left[ f\left( S_{T}\right)  \right]$, which exhibits often no analytic solution. The Monte-Carlo method can be employed by drawing randomly paths of the corresponding SDEs.

"""

# ╔═╡ a5fb2e66-8f34-11eb-21f8-4777948dbc51
md"""## 1) Initialize a Vector to store the Stock prices

"""

# ╔═╡ 6d6ccc96-8f43-11eb-374e-c13976ab9d14
function initialize_s(M)
	S=zeros(1,M)
	return S
end

# ╔═╡ 0244da6c-8f48-11eb-18ff-25b236012972
md"""## 2)Euler-Maruyama Method

"""

# ╔═╡ 02b8706a-8f48-11eb-2e8a-69544a63d964
function E_M!(S,s0,M,N,r,t,sigma,T)
	clf()
	PyPlot.gcf().set_size_inches(16,8)
	for i=1:M
		s_0=s0
		ss=zeros(N,1)
		for k=1:N
			s=s_0+r*t*s_0+s_0*sigma*sqrt(t)*randn()
			s_0=s
			ss[k]=s_0
		end
		plot(t:t:T,ss,linewidth=0.9)
		S[i]=s_0
	end
	figure=gcf()
end

# ╔═╡ 031a8930-8f48-11eb-01be-05f1f317f424
md"""## 3)Stochastic Runge-Kuntta Method

"""

# ╔═╡ 051fd41e-8f49-11eb-1fb8-6b8fdb7daf5d
function R_K!(S,s0,M,N,r,t,sigma,T)
	clf()
	PyPlot.gcf().set_size_inches(16,8)
	for i=1:M
		s_0=s0
		ss=zeros(N,1)
		for k=1:N
			s_hat=s_0+r*s_0*t+s_0*sigma*sqrt(t)
			s=s_0+r*s_0*t+sigma*s_0*sqrt(t)*randn()+(sigma*s_hat-sigma*s_0)*((sqrt(t)*randn())^2-t)/(2*sqrt(t))
			s_0=s
			ss[k]=s_0
		end
		plot(t:t:T,ss,linewidth=0.9)
		S[i]=s_0
	end
	figure=gcf()
end

# ╔═╡ e4a8a43e-8f4a-11eb-20b2-c780f5a5145c
md"""## 4)Calculate the Option Price at  $t_0$
"""

# ╔═╡ 01c32f1c-8f4b-11eb-3e15-073cab370c42
function cal_v(S,K,r,T)
	v=max.(S.-K,0)
	return mean(v)*exp(-r*T)
end

# ╔═╡ 51b63da8-8f4a-11eb-1207-4bdddb77508f
md"""## 5)Implement
"""

# ╔═╡ a7785956-8f4a-11eb-25d4-1be93931609c
M=100000;

# ╔═╡ b7a71d9c-8f4a-11eb-1d74-09aa09de6e70
S1=initialize_s(M)

# ╔═╡ b83a40ea-8f4a-11eb-1378-355ce7509b16
E_M!(S1,s0,M,N,r,t,sigma,T)

# ╔═╡ 3d6d4c14-8f4b-11eb-20d8-13199f21542e
cal_v(S1,K,r,T)

# ╔═╡ 6412f0c0-8f4c-11eb-386a-7fa8a85fae33
S2=initialize_s(M)

# ╔═╡ 68bd15b6-8f4b-11eb-0fed-fb53758b2f27
R_K!(S2,s0,M,N,r,t,sigma,T)

# ╔═╡ 694a2b52-8f4b-11eb-1f4b-4fac2006920e
cal_v(S2,K,r,T)

# ╔═╡ Cell order:
# ╟─07abc4fa-8f34-11eb-0f33-636a1c8257b1
# ╟─a5fb2e66-8f34-11eb-21f8-4777948dbc51
# ╠═6d6ccc96-8f43-11eb-374e-c13976ab9d14
# ╟─0244da6c-8f48-11eb-18ff-25b236012972
# ╠═02b8706a-8f48-11eb-2e8a-69544a63d964
# ╠═031a8930-8f48-11eb-01be-05f1f317f424
# ╠═051fd41e-8f49-11eb-1fb8-6b8fdb7daf5d
# ╠═e4a8a43e-8f4a-11eb-20b2-c780f5a5145c
# ╠═01c32f1c-8f4b-11eb-3e15-073cab370c42
# ╟─51b63da8-8f4a-11eb-1207-4bdddb77508f
# ╠═812f50f0-9c84-11eb-278d-b552fcf958e4
# ╠═8eb6c190-9c84-11eb-37de-7f99b0e41e5c
# ╠═5f8df1c8-8f4a-11eb-2aaa-a743941f3de9
# ╠═a7785956-8f4a-11eb-25d4-1be93931609c
# ╠═b7a71d9c-8f4a-11eb-1d74-09aa09de6e70
# ╠═b83a40ea-8f4a-11eb-1378-355ce7509b16
# ╠═3d6d4c14-8f4b-11eb-20d8-13199f21542e
# ╠═6412f0c0-8f4c-11eb-386a-7fa8a85fae33
# ╠═68bd15b6-8f4b-11eb-0fed-fb53758b2f27
# ╠═694a2b52-8f4b-11eb-1f4b-4fac2006920e
