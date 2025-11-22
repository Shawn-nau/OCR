class JointLikelyhood(nn.Module):
    def __init__(self,args,device):
        super(JointLikelyhood, self).__init__()
        
    def nb_cdf_sum(self,y, r, p):
        """
        Compute NB CDF by brute‐force summing PMFs up to each y[i].
        y: (batch,) integer tensor
        r: (batch,) total_count
        p: (batch,) success_prob
        returns (batch,) tensor of CDFs
        """
        device = y.device
        max_y = int(y.max().item())
        k     = torch.arange(max_y + 1, device=device)          # shape (max_y+1,)

        # Expand params to (batch,1) so log_prob(k) → (batch, max_y+1)
        r2 = r.unsqueeze(1)   # (batch,1)
        p2 = p.unsqueeze(1)   # (batch,1)
        nb = torch.distributions.negative_binomial.NegativeBinomial(total_count=r2, logits=p2)

        # k.unsqueeze(0): (1, max_y+1) broadcasts with (batch,1) → (batch, max_y+1)
        logpmf = nb.log_prob(k.unsqueeze(0))  # (batch, max_y+1)
        pmf    = logpmf.exp()

        # mask out terms above y[i]
        mask = (k.unsqueeze(0) <= y.unsqueeze(1))  # (batch, max_y+1)
        cdf  = (pmf * mask).sum(dim=1)             # (batch,)

        return cdf

    def bivariate_nb_loss_bruteforce(self,y1, y2, r1, p1, r2, p2, rho, eps=1e-6):
        # 1) marginals
        nb1 = torch.distributions.negative_binomial.NegativeBinomial(total_count=r1, logits=p1)
        nb2 = torch.distributions.negative_binomial.NegativeBinomial(total_count=r2, logits=p2)
        logp1 = nb1.log_prob(y1)
        logp2 = nb2.log_prob(y2)

        # 2) CDF by summation (fixed)
        u1 = self.nb_cdf_sum(y1, r1, p1).clamp(eps, 1 - eps)
        u2 = self.nb_cdf_sum(y2, r2, p2).clamp(eps, 1 - eps)

        # 3) probit transform → Gaussian quantiles
        norm = Normal(0., 1.)
        z1 = norm.icdf(u1)
        z2 = norm.icdf(u2)
        
        rho = torch.clamp(rho, min=-0.9999, max=0.9999)
        
        # 4) Gaussian copula log‑density
        Sigma = torch.stack([
            torch.stack([torch.ones_like(rho), rho], dim=-1),
            torch.stack([rho, torch.ones_like(rho)], dim=-1)
        ], dim=-2)  # (batch,2,2)

        mvn = MultivariateNormal(
            loc=torch.zeros_like(rho).unsqueeze(-1).expand(-1,2),
            covariance_matrix=Sigma
        )
        z = torch.stack([z1, z2], dim=-1)
        logc = mvn.log_prob(z) - norm.log_prob(z1) - norm.log_prob(z2)

        # 5) joint log‑likelihood
        ll = logp1 + logp2 + 0.*logc 
        return -ll.mean()
   
    
    def forward(self, r,p, target):           
        rho = F.tanh(p[:,1])  # Map parameter to [-1, 1] range
        return self.bivariate_nb_loss_bruteforce(y1=target[:,0], y2=target[:,1], r1=r[:,0], p1=p[:,0], r2=r[:,1], p2=p[:,0], rho=rho)