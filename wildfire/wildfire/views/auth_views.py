from flask import Blueprint, render_template, request, redirect, url_for, flash, session, g
from werkzeug.security import generate_password_hash, check_password_hash

from wildfire import db
from ..forms import UserCreateForm, UserLoginForm
from ..models import Member

bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    로그인: 아이디/비번 검증 후 세션에 user_username 저장
    """
    form = UserLoginForm()
    if form.validate_on_submit():
        user = Member.query.filter_by(username=form.username.data).first()
        if not user:
            flash('존재하지 않는 아이디입니다. 회원가입이 필요합니다.', 'danger')
        elif not check_password_hash(user.password, form.password.data):
            flash('아이디 또는 비밀번호가 올바르지 않습니다.', 'danger')
        else:
            session.clear()
            session['user_username'] = user.username
            flash('로그인되었습니다.', 'success')
            return redirect(url_for('main.index'))
    return render_template('auth/login.html', form=form)


@bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """
    회원가입: 중복 아이디/이메일 체크 후 생성
    일반 회원가입은 기본적으로 is_admin=False
    """
    form = UserCreateForm()
    if form.validate_on_submit():
        if Member.query.filter_by(username=form.username.data).first():
            flash('이미 존재하는 아이디입니다.', 'danger')
        elif Member.query.filter_by(email=form.email.data).first():
            flash('이미 사용 중인 이메일입니다.', 'danger')
        else:
            new_user = Member(
                username=form.username.data,
                password=generate_password_hash(form.password1.data),
                email=form.email.data,
                name=form.name.data if hasattr(form, "name") else None,
                is_admin=False  # 기본은 관리자 아님
            )
            db.session.add(new_user)
            db.session.commit()
            flash('회원가입이 완료되었습니다. 로그인해 주세요.', 'success')
            return redirect(url_for('auth.signup_success'))
    return render_template('auth/signup.html', form=form)


@bp.route('/signup/success')
def signup_success():
    """
    회원가입 성공 안내 페이지
    """
    return render_template('auth/signup_success.html')


@bp.route('/logout')
def logout():
    """
    로그아웃: 세션 초기화
    """
    session.clear()
    flash('로그아웃되었습니다.', 'info')
    return redirect(url_for('main.index'))


@bp.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    """
    비밀번호 재설정:
      - 사용자 본인확인(아이디 + 이메일)
      - 새 비밀번호/확인 일치 확인
    """
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        new_password = request.form.get('newPassword', '')
        confirm_password = request.form.get('confirmPassword', '')

        if not username or not email or not new_password or not confirm_password:
            flash('모든 필드를 입력해 주세요.', 'danger')
            return redirect(url_for('auth.reset_password'))

        if new_password != confirm_password:
            flash('비밀번호가 일치하지 않습니다.', 'danger')
            return redirect(url_for('auth.reset_password'))

        user = Member.query.filter_by(username=username, email=email).first()
        if not user:
            flash('해당 사용자 정보를 찾을 수 없습니다.', 'danger')
            return redirect(url_for('auth.reset_password'))

        user.password = generate_password_hash(new_password)
        db.session.commit()

        flash('비밀번호가 재설정되었습니다. 다시 로그인해 주세요.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/reset_pw.html')


@bp.route('/find-id', methods=['GET', 'POST'])
def find_id():
    """
    아이디 찾기:
      - 이름 + 이메일로 조회하여 username 안내
    """
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()

        if not name or not email:
            flash('이름과 이메일을 모두 입력해 주세요.', 'danger')
            return redirect(url_for('auth.find_id'))

        user = Member.query.filter_by(name=name, email=email).first()

        if user:
            flash(f'회원님의 아이디는: {user.username} 입니다.', 'info')
        else:
            flash('입력한 정보와 일치하는 아이디가 없습니다.', 'danger')

        return redirect(url_for('auth.find_id'))

    return render_template('auth/find_id.html')


# ✅ 앱 전체 요청 전에 현재 로그인 사용자 로딩
@bp.before_app_request
def load_logged_in_user():
    """
    - 세션에서 user_username 읽어 DB 조회
    - g.user에 Member 또는 None 저장
    - 템플릿/뷰에서 g.user와 g.user.is_admin 사용 가능
    """
    user_username = session.get('user_username')
    if user_username is None:
        g.user = None
    else:
        g.user = Member.query.get(user_username)