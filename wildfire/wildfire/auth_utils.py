from functools import wraps
from flask import g, redirect, url_for, abort

def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not getattr(g, "user", None):
            # 로그인 페이지가 있으면 redirect로 바꿔도 됩니다.
            return redirect(url_for("member.login"))
        if not getattr(g.user, "is_admin", False):
            return abort(403)
        return view(*args, **kwargs)
    return wrapped