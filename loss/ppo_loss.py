import torch

def clip_ppo_loss(
        newlogprob,
        entropy,
        value,
        b_values,
        b_logprob,
        b_advantages,
        b_returns,
        clip_coef,
        ent_coef,
        vf_coef
    ):
    logratio = newlogprob - b_logprob
    ratio = logratio.exp()

    with torch.no_grad():

        # normalize advantaegs
        advantages = b_advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(
            ratio, 1 - clip_coef, 1 + clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        value = value.flatten()
        v_loss_unclipped = (value - b_returns) ** 2
        v_clipped = b_values + torch.clamp(
                value - b_values,
                -clip_coef,
                clip_coef,
            )
        v_loss_clipped = (v_clipped - b_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

    return loss


def clip_hippo_loss(
        m_newlogprob,
        m_entropy,
        m_value,
        b_manager_entropy,
        b_manager_advantages,
        b_manager_values,
        b_manager_returns,
        s_newlogprob,
        s_entropy,
        s_value,
        b_skill_entropy,
        b_skill_advantages,
        b_skill_values,
        b_skill_returns,
        clip_coef,
        ent_coef,
        vf_coef
    ):
    '''
    # normalize advantages
    m_advantages = b_manager_advantages
    print(torch.std(m_advantages))
    m_advantages = (m_advantages - torch.mean(m_advantages)) / (
                    torch.std(m_advantages) + 1e-8
                )
    s_advantages = b_skill_advantages
    print(torch.std(s_advantages))
    s_advantages = (s_advantages - torch.mean(s_advantages)) / (
                    torch.std(s_advantages) + 1e-8
                )

    print(m_advantages)
    print(s_advantages)
    '''

    # Manager loss
    '''
    print("1", b_manager_entropy)
    print("1-2", m_entropy)
    print("1-3", torch.log(m_entropy))
    print("1-4", torch.log(b_manager_entropy))
    '''
    m_lr = torch.exp(torch.log(b_manager_entropy) - torch.log(m_entropy))
    manager_loss = torch.minimum(m_lr * b_manager_advantages, torch.clip(m_lr, 1 - clip_coef, 1 + clip_coef) * b_manager_advantages)
    manager_loss = -torch.mean(manager_loss)
    #print(manager_loss)

    m_value = m_value.flatten()
    m_v_loss_unclipped = (m_value - b_manager_returns) ** 2
    m_v_clipped = b_manager_values + torch.clamp(m_value - b_manager_values, -clip_coef, clip_coef,)
    m_v_loss_clipped = (m_v_clipped - b_manager_returns) ** 2
    m_v_loss_max = torch.max(m_v_loss_unclipped, m_v_loss_clipped)
    m_v_loss = 0.5 * m_v_loss_max.mean()
    
    manager_loss = manager_loss + vf_coef * m_v_loss

    # skills loss
    '''
    print("2", b_skill_entropy)
    print("2-2", s_entropy)
    print("2-3", torch.log(s_entropy))
    print("2-4", torch.log(b_skill_entropy))
    '''
    s_lr = torch.exp(torch.log(b_skill_entropy) - torch.log(s_entropy))
    skill_loss = torch.minimum(s_lr * b_skill_advantages, torch.clip(s_lr, 1 - clip_coef, 1 + clip_coef) * b_skill_advantages)
    skill_loss = -torch.mean(skill_loss)
    #print(skill_loss)
    
    s_value = s_value.flatten()
    s_v_loss_unclipped = (s_value - b_skill_returns) ** 2
    s_v_clipped = b_skill_values + torch.clamp(s_value - b_skill_values, -clip_coef, clip_coef,)
    s_v_loss_clipped = (s_v_clipped - b_skill_returns) ** 2
    s_v_loss_max = torch.max(s_v_loss_unclipped, s_v_loss_clipped)
    s_v_loss = 0.5 * s_v_loss_max.mean()

    skill_loss = skill_loss + s_v_loss * vf_coef

    surr_loss = manager_loss + skill_loss

    return surr_loss, manager_loss, skill_loss