from fastai.vision.all import *
import gradio as gr
from PIL import Image
import numpy as np
import timm

# Load the learner
learn = load_learner('Minerals_Convnext_large.pkl')



# Define the categories and corresponding paragraphs
Categories = ['Amphibolite', 'acanthite', 'apatite', 'azurite', 'baryte', 'beryl', 'biotite', 'bornite', 'calcite', 'carbonaceous phyllite', 'carbonaceous schists', 'carbonaceous schists gnesis', 'cerussite', 'chrysocolla', 'copper', 'corundum', 'fluorite', 'gypsum', 'hematite', 'malachite', 'muscovite', 'orthoclase', 'pyrite', 'pyromorphite', 'quartz', 'smithsonite', 'talc', 'topaz', 'wulfenite']

Paragraphs = {
    'Amphibolite': "Amphibolite is a dark, heavy, metamorphic rock primarily composed of amphibole minerals and plagioclase feldspar, often exhibiting a banded texture. It forms under high-pressure and high-temperature conditions, typically in regions with significant tectonic activity. Amphibolite is commonly found in mountain ranges and is valued for its durability, often used as a construction material or decorative stone in landscaping. It is generally not rare and is often sold for $10 to $50 per ton, depending on quality and location.",
    'Acanthite': "Acanthite is a silver sulfide mineral (Ag₂S) that typically forms in a monoclinic crystal system. It is an important ore of silver and is known for its metallic luster and dark gray color. Acanthite is often found in hydrothermal veins and is valued in the mineral collecting community for its rarity. Prices can range from $5 to $100 per ounce, depending on purity and market demand.",
    'apatite': "Apatite is a group of phosphate minerals that come in a variety of colors, including green, blue, and brown. It is commonly found in igneous rocks and is an essential mineral for producing phosphate fertilizers. Apatite is relatively common and affordable, often used as a gemstone.",
    'azurite': "Azurite is a deep blue copper mineral often found in association with malachite. It forms in the oxidized zones of copper ore deposits and is valued for its vibrant color, making it popular among collectors. Azurite is relatively rare and can be quite expensive, especially in high-quality specimens.",
    'baryte': "Baryte, or barite, is a barium sulfate mineral that commonly forms in sedimentary rocks, especially in limestone. It is used extensively in the oil and gas industry as a weighting agent in drilling fluids. Baryte is abundant and generally inexpensive.",
    'beryl': "Beryl is a beryllium aluminum silicate mineral that comes in several color varieties, including emerald (green) and aquamarine (blue). It's found in granitic pegmatites and is highly valued as a gemstone. Beryl can range from common to extremely rare, depending on the color and clarity.",
    'biotite': "Biotite is a common phyllosilicate mineral within the mica group, known for its dark brown to black color. It typically occurs in igneous and metamorphic rocks. Biotite is widely available and relatively inexpensive, often used in geological studies.",
    'bornite': "Bornite, also known as peacock ore, is a copper iron sulfide mineral with a bronze to purple iridescence. It is a valuable copper ore found in hydrothermal veins. Bornite is moderately rare and can fetch high prices among collectors due to its colorful appearance.",
    'calcite': "Calcite is a widespread carbonate mineral that forms in a variety of geological environments. It's recognized for its rhombohedral crystal form and reacts with acid by effervescing. Calcite is common and inexpensive, used in the production of cement and lime.",
    'carbonaceous phyllite': "Carbonaceous phyllite is a type of low-grade metamorphic rock rich in carbonaceous material, giving it a dark color. It forms under moderate pressure and temperature conditions and is commonly found in orogenic belts. Its rarity and value are moderate, depending on its specific carbon content.",
    'carbonaceous schists': "Carbonaceous schists are metamorphic rocks that are rich in carbonaceous material, giving them a distinct dark color. They form under high pressure and temperature, often in regions of significant tectonic activity. These rocks are relatively rare and can be of interest to geologists for their unique formation history.",
    'carbonaceous schists gnesis': "Carbonaceous schists gneiss is a high-grade metamorphic rock characterized by a banded appearance and significant carbonaceous content. It forms under intense heat and pressure conditions, typically in deep crustal environments. This rock type is rare and often studied for its complex geological history.",
    'cerussite': "Cerussite is a lead carbonate mineral that forms in the oxidized zones of lead ore deposits. It's known for its high density and bright luster, often used as a lead ore. Cerussite is moderately rare and can be quite valuable, especially in well-formed crystals.",
    'chrysocolla': "Chrysocolla is a hydrated copper silicate mineral known for its bright blue-green color. It forms in the oxidized zones of copper deposits and is often used as a gemstone. Chrysocolla is moderately rare and can vary in price depending on the quality of the specimen.",
    'copper': "Copper is a native element and metal that is highly conductive and malleable. It is found in various geological settings, often in association with other copper minerals like azurite and bornite. Copper is relatively abundant and widely used in electrical wiring and other applications.",
    'corundum': "Corundum is a crystalline form of aluminum oxide and is one of the hardest minerals, second only to diamond. It comes in various colors, with red corundum known as ruby and blue as sapphire. Corundum is valuable and can be extremely rare depending on color and clarity.",
    'fluorite': "Fluorite is a colorful mineral composed of calcium fluoride, often forming in hydrothermal veins. It is known for its cubic crystals and is used in the production of fluorine and as a flux in steelmaking. Fluorite is moderately common and relatively affordable, though high-quality specimens can be valuable.",
    'gypsum': "Gypsum is a soft sulfate mineral widely used in construction, especially in the production of plaster and drywall. It forms in evaporite deposits and is known for its flexibility and ability to form in large crystals. Gypsum is abundant and inexpensive.",
    'hematite': "Hematite is an iron oxide mineral that is the primary ore of iron. It forms in various geological environments and is recognized for its metallic luster and reddish streak. Hematite is common and widely used in the steel industry, making it relatively affordable.",
    'malachite': "Malachite is a copper carbonate mineral known for its vivid green color and banded patterns. It forms in the oxidized zones of copper deposits and is often used as a gemstone or decorative stone. Malachite is moderately rare and can be quite expensive, especially in high-quality specimens.",
    'muscovite': "Muscovite is a common mica mineral that is highly flexible and can be split into thin sheets. It forms in igneous and metamorphic rocks and is used in the electrical and construction industries. Muscovite is abundant and inexpensive.",
    'orthoclase': "Orthoclase is a feldspar mineral that commonly forms in igneous rocks like granite. It is known for its pink to white color and is used in the production of ceramics and glass. Orthoclase is relatively common and affordable.",
    'pyrite': "Pyrite, also known as fool's gold, is an iron sulfide mineral with a metallic luster and pale brass-yellow hue. It forms in various geological environments and is often found in association with gold. Pyrite is common and relatively inexpensive, though it's prized for its aesthetic appeal.",
    'pyromorphite': "Pyromorphite is a lead chloride phosphate mineral often found in the oxidized zones of lead deposits. It is known for its bright green to yellow color and barrel-shaped crystals. Pyromorphite is moderately rare and can be quite valuable among mineral collectors.",
    'quartz': "Quartz is one of the most abundant minerals in the Earth's crust, composed of silicon dioxide. It comes in many varieties, including amethyst and citrine, and is used in a wide range of applications, from jewelry to electronics. Quartz is common and generally affordable, though rare varieties can be expensive.",
    'smithsonite': "Smithsonite is a zinc carbonate mineral that forms in the oxidized zones of zinc deposits. It is often found in botryoidal forms with a range of colors, including blue, green, and pink. Smithsonite is moderately rare and can be valuable, especially in well-formed specimens.",
    'talc': "Talc is a soft mineral composed of magnesium silicate, often used in the production of talcum powder. It forms in metamorphic rocks and is known for its greasy feel and softness. Talc is abundant and inexpensive.",
    'topaz': "Topaz is a silicate mineral that comes in a variety of colors, including blue, yellow, and pink. It forms in granitic pegmatites and is highly valued as a gemstone. Topaz can range from common to extremely rare, depending on the color and clarity.",
    'wulfenite': "Wulfenite is a lead molybdate mineral known for its bright orange to red crystals. It forms in the oxidized zones of lead deposits and is highly prized by collectors. Wulfenite is relatively rare and can be quite expensive, especially in high-quality specimens."
}



def classify_image(img):
    # Convert to PIL Image if not already
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.resize((192, 192))  # Resize the image to the required size
    pred, idx, prob = learn.predict(img)  # Predict using the model

    # Ensure prob is a numpy array
    prob = np.array(prob)

    # Get the top 2 categories with highest probabilities
    top_2_idx = prob.argsort()[-2:][::-1]  # Sort indices and select top 2
    top_2_categories = [Categories[i] for i in top_2_idx]
    top_2_paragraphs = [Paragraphs[category] for category in top_2_categories]

    # Prepare the output
    label_output = {category: float(prob[i]) for category, i in zip(top_2_categories, top_2_idx)}
    paragraphs_output = "\n\n".join(top_2_paragraphs)

    return label_output, paragraphs_output


# Setup Gradio interface
title = gr.Markdown("# Minerals-Crystal Identifier")
image = gr.Image()
label = gr.Label()
paragraphs = gr.Textbox(lines=10, label="Minerals Facts")
examples = [["Biotite.jpg"], ["quartz.jpg"],["malachite.jpg"],["pyromorphite2.jpg"],["fluorite.jpg"]]  

intf = gr.Interface(fn=classify_image, inputs=image, outputs=[label, paragraphs], examples=examples, flagging_dir="/tmp/flagged")
intf.launch(inline=False)