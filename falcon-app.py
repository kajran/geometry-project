# # Run Falcon-40B with AutoGPTQ
#
# In this example, we run a quantized 4-bit version of Falcon-40B, the first open-source large language
# model of its size, using HuggingFace's [transformers](https://huggingface.co/docs/transformers/index)
# library and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ).
#
# Due to the current limitations of the library, the inference speed is a little under 1 token/second and the
# cold start time on Modal is around 25s.
#
# For faster inference at the expense of a slower cold start, check out
# [Running Falcon-40B with `bitsandbytes` quantization](/docs/guide/ex/falcon_bitsandbytes). You can also
# run a smaller, 7-billion-parameter model with the [OpenLLaMa example](/docs/guide/ex/openllama).
#
# ## Setup
#
# First we import the components we need from `modal`.

from pathlib import Path

from modal import Image, Stub, gpu, method, web_endpoint, asgi_app, Mount

# ## Define a container image
#
# To take advantage of Modal's blazing fast cold-start times, we download model weights
# into a folder inside our container image. These weights come from a quantized model
# found on Huggingface.
IMAGE_MODEL_DIR = "/model"


def download_model():
    from huggingface_hub import snapshot_download

    model_name = "TheBloke/falcon-7b-instruct-GPTQ"
    snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR)


# Now, we define our image. We'll use the `debian-slim` base image, and install the dependencies we need
# using [`pip_install`](/docs/reference/modal.Image#pip_install). At the end, we'll use
# [`run_function`](/docs/guide/custom-container#running-a-function-as-a-build-step-beta) to run the
# function defined above as part of the image build.

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "auto-gptq @ git+https://github.com/PanQiWei/AutoGPTQ.git@b5db750c00e5f3f195382068433a3408ec3e8f3c",
        "einops==0.6.1",
        "hf-transfer~=0.1",
        "huggingface_hub==0.14.1",
        "transformers @ git+https://github.com/huggingface/transformers.git@f49a3453caa6fe606bb31c571423f72264152fce",
    )
    # Use huggingface's hi-perf hf-transfer library to download this large model.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model)
)

# Let's instantiate and name our [Stub](/docs/guide/apps).
stub = Stub(name="example-falcon-gptq", image=image)


# ## The model class
#
# Next, we write the model code. We want Modal to load the model into memory just once every time a container starts up,
# so we use [class syntax](/docs/guide/lifecycle-functions) and the `__enter__` method.
#
# Within the [@stub.cls](/docs/reference/modal.Stub#cls) decorator, we use the [gpu parameter](/docs/guide/gpu)
# to specify that we want to run our function on an [A100 GPU](/pricing). We also allow each call 10 mintues to complete,
# and request the runner to stay live for 5 minutes after its last request.
#
# The rest is just using the `transformers` library to run the model. Refer to the
# [documentation](https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationMixin.generate)
# for more parameters and tuning.
#
# Note that we need to create a separate thread to call the `generate` function because we need to
# yield the text back from the streamer in the main thread. This is an idiosyncrasy with streaming in `transformers`.
@stub.cls(gpu=gpu.A100(count=2), timeout=60 * 10, container_idle_timeout=60 * 5)
class Falcon40BGPTQ:
    def __enter__(self):
        from auto_gptq import AutoGPTQForCausalLM
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            IMAGE_MODEL_DIR, use_fast=True
        )
        print("Loaded tokenizer.")

        self.model = AutoGPTQForCausalLM.from_quantized(
            IMAGE_MODEL_DIR,
            trust_remote_code=True,
            use_safetensors=True,
            device_map="auto",
            use_triton=False,
            strict=False,
        )
        print("Loaded model.")

    @method()
    def generate(self, prompt: str):
        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_special_tokens=True
        )
        generation_kwargs = dict(
            inputs=inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask,
            temperature=0.1,
            max_new_tokens=512,
            streamer=streamer,
        )

        # Run generation on separate thread to enable response streaming.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text

        thread.join()


# ## Run the model
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run -q falcon_gptq.py`. The `-q` flag
# enables streaming to work in the terminal output.
# prompt_template = (
#     "A chat between a curious human user and an artificial intelligence assistant. The assistant give a helpful, detailed, and accurate answer to the user's question."
#     "\n\nUser:\n{}\n\nAssistant:\n"
# )
prompt_template = (
    """Given a triangle $ABC$ satisfying $AC+BC=3\cdot AB$. The incircle of triangle $ABC$ has center $I$ and touches the sides $BC$ and $CA$ at the points $D$ and $E$, respectively. Let $K$ and $L$ be the reflections of the points $D$ and $E$ with respect to $I$. Prove that the points $A$, $B$, $K$, $L$ lie on one circle. => pair B = origin;
pair C = (4,0);
pair A = 3*dir(35.430945);

pair I = incenter(A,B,C);
pair D = foot(I,B,C);
pair E = foot(I,A,B);

pair K = 2*I-D;
pair L = 2*I-E;

draw(A--B--C--cycle);
draw(incircle(A, B, C), grey);

dot("$A$",A,dir(90));
dot("$B$",B,dir(200));
dot("$C$",C,dir(-40));

dot("$I$",I,dir(120));
dot("$D$",D,dir(-90));
dot("$E$",E,dir(120));

dot("$K$",K,S);
dot("$L$",L,dir(90));



Let $ABCD$ be a parallelogram. A variable line $g$ through the vertex $A$ intersects the rays $BC$ and $DC$ at the points $X$ and $Y$, respectively. Let $K$ and $L$ be the $A$-excenters of the triangles $ABX$ and $ADY$. Show that the angle $\measuredangle KCL$ is independent of the line $g$. => pair excenter(pair A=(0,0), pair B=(0,0), pair C=(0,0))
{{
    pair P,Q;
    P=rotate(-1*((angle(A-B)-angle(C-B))*90/pi+90),B)*A;
    Q=rotate((angle(B-C)-angle(A-C))*90/pi+90,C)*A;
    return extension(B,P,C,Q);
}}

pair A = (0.8,3);
pair D = origin;
pair C = (4,0);
pair B = A + C;
pair X = extension(B,C,A,A+dir(-55));
pair Y = extension(D,C,A,A+dir(-55));

pair K = excenter(A,X,B);
pair L = excenter(A,D,Y);


draw(A--2*X-A);
draw(A--2*D-A);
draw(A--2*B-A);
draw(A--K);

dot(A^^B^^C^^D^^K^^L^^X^^Y);
label("$A$",A,dir(120));
label("$B$",B,dir(30));
label("$C$",C,dir(-50));
label("$D$",D,dir(-160));
label("$Y$",Y,dir(70));
label("$K$",K,E);
label("$L$",L,W);
label("$X$",X,dir(100));



Let $\triangle ABC$ be an acute-angled triangle with $AB \not= AC$. Let $H$ be the orthocenter of triangle $ABC$, and let $M$ be the midpoint of the side $BC$. Let $D$ be a point on the side $AB$ and $E$ a point on the side $AC$ such that $AE=AD$ and the points $D$, $H$, $E$ are on the same line. Prove that the line $HM$ is perpendicular to the common chord of the circumscribed circles of triangle $\triangle ABC$ and triangle $\triangle ADE$. => pair A = 1.5*dir(70);
pair B = origin;
pair C = (2,0);
pair M = (B+C)/2;

pair H = orthocenter(A,B,C);
pair D = point(A--B,intersections(A--B,H,K)[0]);
pair E = point(A--C,intersections(A--C,H,K)[0]);

draw(A--B--C--cycle);
draw(D--E, grey);
draw(H--M, grey);

dot("$A$",A,dir(90));
dot("$B$",B,dir(200));
dot("$C$",C,dir(-40));

dot("$M$",M,dir(-90));
dot("$H$",H,dir(100));
dot("$D$",D,dir(150));
dot("$E$",E,dir(40));



Let $ABC$ be an acute triangle with $D, E, F$ the feet of the altitudes lying on $BC, CA, AB$ respectively. One of the intersection points of the line $EF$ and the circumcircle is $P.$ The lines $BP$ and $DF$ meet at point $Q.$ Prove that $AP = AQ.$ => real markscalefactor = 0.05;
path rightanglemark(pair A, pair B, pair C, real s=8)
{{
    pair P,Q,R;
    P=s*markscalefactor*unit(A-B)+B;
    R=s*markscalefactor*unit(C-B)+B;
    Q=P+R-B;
    return P--Q--R;
}}


pair C = (14,0);
pair A = (9,12);
pair B = (0,0);
pair D = foot(A,B,C);
pair E = foot(B,A,C);
pair F = foot(C,A,B);
pair om = extension(E,F,C,B);
pair nom = extension(E,F,D,(5,3));
path circ = circumcircle(A,B,C);

pair P1 = intersectionpoint(F--om, circ);
pair P2 = intersectionpoint(E--nom, circ);
pair Q1 = intersectionpoint(D--F, B--P1);
pair Q2 = extension(B,P2,D,F);
pair Ep = reflect(A,B)*E;
pair Cp = reflect(A,B)*C;
pair Hp = reflect(A,B)*H;
pair Dp = reflect(A,B)*D;
draw(A--B--C--A--D--F--C);
draw(B--E);
draw(P1--P2);
draw(F--Q2);
draw(P1--B--Q2, dotted);
dot(A^^B^^C^^D^^E^^F^^Q1^^Q2^^P1^^P2);
draw(circ);
markscalefactor=0.05;
draw(rightanglemark(C,F,A));
draw(rightanglemark(B,E,C));
draw(rightanglemark(A,D,B));
label("$A$", A, dir(90));
label("$B$", B, dir(225));
label("$C$", C, dir(315));
label("$D$", D, dir(-90));
label("$E$", E, dir(45));
label("$F$", F, 1.2*dir(200));
label("$P_1$", P1, dir(0));
label("$P_2$", P2, dir(135));
label("$Q_1$", Q1, dir(250));
label("$Q_2$", Q2, dir(90));



Let ABC be a triangle, and let $D$, $E$, and $F$ denote the feet of the altitudes from $A$, $B$, and $C$, respectively. Then, $\triangle DEF$ is called the orthic triangle of $\triangle ABC$. => pair A = dir(110);
pair B = dir(210);
pair C = dir(330);
draw(A--B--C--cycle);

dot("$A$", A, dir(110));
dot("$B$", B, B);
dot("$C$", C, C);

pair H = A+B+C;
dot("$H$", H, dir(100));

pair D = foot(A, B, C);
pair E = foot(B, C, A);
pair F = foot(C, A, B);

dot("$D$", D, dir(-90));
dot("$E$", E, dir(50));
dot("$F$", F, dir(110));

draw(A--D);
draw(B--E);
draw(C--F);



Points A, B, C, D, E lie on a circle ω and point P lies outside the circle. The given points are such that (i) lines PB and P D are tangent to ω, (ii) P, A, C are collinear, and (iii) DE is parallel to AC. Prove that BE bisects AC. => B = dir 100
D = dir 210
E = dir 330
P = 2*B*D/(B+D)
A = IP P--(P+8*(E-D)) unitcircle
M = extension B E A P R45
C = 2*M-A R100

unitcircle 0.1 lightblue / lightblue
P--A lightblue
B--P--D lightblue
D--E heavygreen
A--E--C heavygreen

B--A--D--C--cycle heavycyan



Let ABC be a triangle. The incircle of $\triangle ABC$ is tangent to AB at AC and D and E respectively. Let O denote the circumcenter of $\triangle BCI$. Prove that $\angle ODB = \angle OEC$. => A = dir 110
B = dir 210
C = dir 330
A--B--C--cycle 0.1 lightcyan / lightblue
O = dir -90
I = incenter A B C R90
D = foot I A B
E = foot I A C
incircle A B C dashed lightblue
B--I--C lightblue
D--O--E heavygreen
"""
"\n\n\n{} => "
)

@stub.local_entrypoint()
def cli():
    # question = "What are the main differences between Python and JavaScript programming languages?"
    # question = "Let $AXYZB$ be a convex pentagon inscribed in a semicircle of diameter $AB$. Denote by $P, Q, R, S$ the feet of the perpendiculars from $Y$ onto lines $AX, BX, AZ, BZ$, respectively. Prove that the acute angle formed by lines $PQ$ and $RS$ is half the size of $\angle XOZ$, where $O$ is the midpoint of segment $AB$."
    question = "Let ABC be an acute triangle with orthocenter H, and let W be a point on the side BC, between B and C. The points M and N are the feet of the altitudes drawn from B and C, respectively. Suppose ω1 is the circumcircle of triangle BW N and X is a point such that W X is a diameter of ω1. Similarly, ω2 is the circumcircle of triangle CWM and Y is a point such that W Y is a diameter of ω2. Show that the points X, Y , and H are collinear."
    model = Falcon40BGPTQ()
    for text in model.generate.remote_gen(prompt_template.format(question)):
        print(text, end="", flush=True)
    pass


# ## Serve the model
# Finally, we can serve the model from a web endpoint with `modal deploy falcon_gptq.py`. If
# you visit the resulting URL with a question parameter in your URL, you can view the model's
# stream back a response.
# You can try our deployment [here](https://modal-labs--example-falcon-gptq-get.modal.run/?question=Why%20are%20manhole%20covers%20round?).
@stub.function(timeout=60 * 10)
@web_endpoint()
def get(question: str):
    from itertools import chain

    from fastapi.responses import StreamingResponse

    model = Falcon40BGPTQ()
    return StreamingResponse(
        chain(
            ("Loading model. This usually takes around 20s ...\n\n"),
            model.generate.remote_gen(prompt_template.format(question)),
        ),
        media_type="text/event-stream",
    )
    

from os.path import dirname, join

current_dir = dirname(__file__)  # this will be the location of the current .py file
templates_dir = join(current_dir, 'templates')


frontend_path = Path(__file__).parent
print("frontend", frontend_path)

@stub.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    keep_warm=1,
    allow_concurrent_inputs=10,
    timeout=60 * 10,
)
@asgi_app(label="hack-app")
def app():
    # import json

    # import fastapi
    # import fastapi.staticfiles
    # from fastapi.responses import StreamingResponse

    # web_app = fastapi.FastAPI()

    # @web_app.get("/stats")
    # async def stats():
    #     stats = await Model().generate_stream.get_current_stats.aio()
    #     return {
    #         "backlog": stats.backlog,
    #         "num_total_runners": stats.num_total_runners,
    #     }

    # @web_app.get("/completion/{question}")
    # async def completion(question: str):
    #     from urllib.parse import unquote

    #     async def generate():
    #         async for text in Model().generate_stream.remote_gen.aio(
    #             unquote(question)
    #         ):
    #             yield f"data: {json.dumps(dict(text=text), ensure_ascii=False)}\n\n"

    #     return StreamingResponse(generate(), media_type="text/event-stream")

    # web_app.mount(
    #     "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    # )
    # return web_app



    from fastapi import FastAPI, Request, Form
    from fastapi.responses import HTMLResponse
    from fastapi.templating import Jinja2Templates
    from starlette.staticfiles import StaticFiles
    import os

    app = FastAPI()
    templates = Jinja2Templates(directory="/assets/templates")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        # print("index", dirname(__file__))
        # print(os.listdir("/static"))
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/", response_class=HTMLResponse)
    async def capitalize_text(request: Request, user_input: str = Form(...)):
        question = user_input
        model = Falcon40BGPTQ()
        response = ""
        for text in model.generate.remote_gen(prompt_template.format(question)):
            response += text
        return templates.TemplateResponse("index.html", {"request": request, "capitalized_text": response})

    app.mount("/static", StaticFiles(directory="/assets/static"), name="static")
    return app