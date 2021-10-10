import os
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, abort, Response
from flaskblog import app, db, bcrypt
from flaskblog.forms import RegistrationForm, LoginForm, UpdateAccountForm, PostForm, SelectFile,GetDate
from flaskblog.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required
import cv2,subprocess,datetime,json

@app.route("/")
@app.route("/home")
def home():
    posts = Post.query.all()
    return render_template('home.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        #image_file=url_for('default.jpg')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)#,image_file=image_file)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        print("picture data {}".format(form))
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)


@app.route("/post/new", methods=['GET', 'POST'])
@login_required
def new_post():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(title=form.title.data, content=form.content.data, author=current_user)
        db.session.add(post)
        db.session.commit()
        flash('Your note has been created!', 'success')
        return redirect(url_for('home'))
    return render_template('create_post.html', title='New Note',
                           form=form, legend='New Note')


@app.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', title=post.title, post=post)


@app.route("/post/<int:post_id>/update", methods=['GET', 'POST'])
@login_required
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = PostForm()
    if form.validate_on_submit():
        post.title = form.title.data
        post.content = form.content.data
        db.session.commit()
        flash('Your post has been updated!', 'success')
        return redirect(url_for('post', post_id=post.id))
    elif request.method == 'GET':
        form.title.data = post.title
        form.content.data = post.content
    return render_template('create_post.html', title='Update Post',
                           form=form, legend='Update Post')


@app.route("/post/<int:post_id>/delete", methods=['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    flash('Your post has been deleted!', 'success')
    return redirect(url_for('home'))


@app.route("/check_camera", methods=['POST'])
@login_required
def check_camera():
    #print(form.dropdown)
    tag = request.form['Camera Number']
    if tag ==None:
        tag=0
    print(tag)
    cap = cv2.VideoCapture(int(tag))
    if cap is None or not cap.isOpened():
        flash('Camera is not working!', 'danger')
    else:
        flash('Your camera works!', 'success')

    cap.release()
    return redirect(url_for('camera'))

@app.route("/post/start_camera", methods=['POST'])
@login_required
def start_camera():
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        flash('Could not open camera!', 'danger')
    else:
        while(True):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        flash('camera works fine!', 'success')

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('home'))



#Camera on web
video_camera = None
global_frame = None

def video_stream(cam):
    global video_camera
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera(cam)

    while True:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
        #video_camera.__del__()

@app.route('/video_viewer', methods=['POST'])
@login_required
def video_viewer():
    tag = request.form['Camera Number']
    print(tag)
    return Response(video_stream(int(tag)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera', methods=['GET','POST'])
@login_required
def camera():
    #->if video_camera != None:
    #    print("Inside camera del video_camera")
    #->    video_camera.__del__()
        #video_camera=None
    form = SelectFile()
    #return render_template('camera.html', title='Camera',form=form)
    if form.validate_on_submit():#print("picture data {}".format(form.picture))
        form.get_data(form.picture)
        return render_template('layout.html', title='Camera',form=form)

    return render_template('camera.html', title='Camera1',
                               form=form)



@app.route("/post/camera", methods=['POST'])
@login_required
def start_live_surveillance():
    tag = request.form['Camera Number']
    label = request.form['label']

    print("camera number {}".format(tag))
    #post = Post.query.get_or_404(post_id)
    #if post.author != current_user:
    #   abort(403)
    subprocess.call(" python livedetection.py "+str(tag)+" "+(current_user.username)+" "+str(label), shell=True)
    return redirect(url_for('home'))


@app.route('/logs', methods=['GET','POST'])
@login_required
def logs():
    posts=[]
    if request.method == 'POST':
        tag = request.form['date']
        print("value of tag  "+tag)
        #path=url_for('logs', filename=current_user.username+'_'+tag)
        #print(path)
        #with open("logs\"+current_user.username+'_'+tag+".txt","r") as f:
        #    content = f.read()
        #    return Response(f.read(), mimetype='text/plain')
        #return render_template('log.html', content=content)
        #jsonFile = open(path+'.json', 'r')
        #posts = json.load(jsonFile)
        dir=os.getcwd()
        pa=os.path.join(dir,'flaskblog\\logs\\'+current_user.username+'_'+tag+'.json')

        #print(jsonFile)
        if os.path.isfile(pa):
            jsonFile = open(pa,'r')
            data = json.load(jsonFile)
            jsonFile.close()
            print("inside file")
            return render_template('display_logs.html', title='Logs',user=current_user,date_time=datetime.datetime.now(),data=data)
        else:
            return render_template('nofile.html')

        #print(type(tag))
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('log.html', title='Logs',user=current_user,image_file=image_file,date_time=datetime.datetime.now())


class VideoCamera(object):
    def __init__(self,cam=0):
        # Open a camera
        self.cap = cv2.VideoCapture(cam)

        # Initialize video recording environment
        self.is_record = False
        self.out = None

        # Thread for recording
        self.recordingThread = None

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        ret, frame = self.cap.read()

        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)


            return jpeg.tobytes()
        else:
            return None
