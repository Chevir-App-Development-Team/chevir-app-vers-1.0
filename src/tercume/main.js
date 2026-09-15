// /tercume/ səhifəsinin girişi: saytın naviqasiyası + tərcümə sistemi.
import '../styles/base.css'
import '../styles/sections/navbar.css'
import './tercume.css'
import { navbarMarkup } from '../sections/navbar.js'
import './app.js'

document.querySelector('#site-nav').outerHTML = navbarMarkup({ home: false })
