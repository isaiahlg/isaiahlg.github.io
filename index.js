console.log("Welcome to my website!")

const headerTemplate = document.createElement('template');
headerTemplate.innerHTML = `
<style>
        
    .navbar {
        width: 100%;
        background-color: rgb(27, 27, 27);
        overflow: auto;
        opacity: 0.8;
        box-shadow: 0px 10px 20px rgba(255, 255, 255, 0.3);
    }

    .navbar a {
        float: left;
        text-align: center;
        padding-top: 20px;
        padding-bottom: 20px;
        width: 25%;
        color: white;
        text-decoration: none;
        font-size: 14px;
        transition-duration: 0.5s;
    }

    .navbar a:hover {
        background-color: blueviolet;
        font-size: 15px;
        padding-bottom: 19px;
    }
</style>

<div class="navbar">
    <a href="/index.html"><i class="fa fa-fw fa-home"></i> Home</a>
    <a href="/portfolio/home.html"><i class="fa fa-fw fa-robot"></i> Data Science Portfolio</a>
    <a href="/photography/home.html"><i class="fa fa-fw fa-camera"></i> Photography</a>
    <a href="/ultimate/home.html"><i class="fa fa-fw fa-circle"></i> Ultimate</a>
</div>
`

class Header extends HTMLElement {
    constructor() {
        // Always call super first in constructor
        super();
    }

    connectedCallback() {
        const shadowRoot = this.attachShadow({ mode: 'open' });
        shadowRoot.appendChild(headerTemplate.content);
    }
}

customElements.define('navbar-header', Header);