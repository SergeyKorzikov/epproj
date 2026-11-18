import pandas as pd, glob, re
txt='\n'.join(open(f).read() for f in glob.glob('analysis/clean_packs/*.txt'))
je=pd.read_parquet('data/out/messages_je.parquet',columns=['doc_id','body'])
allm=pd.read_parquet('data/out/messages_all.parquet',columns=['doc_id','body'])
checks={
'EFTA02504223':'testosterone','EFTA00819587':'pth','EFTA00818238':'parathyroid','EFTA00981007':'high sugar','EFTA00853059':'metformin',
'EFTA01048400':'metformin','EFTA00554246':'emergency','EFTA02353720':'natural is good','EFTA02642868':'life expectancy','EFTA01026929':'pheromones',
'EFTA01917374':'younger women','EFTA01834862':'design one','EFTA01040296':'CHANGE GAMES','EFTA02466653':'all are games','EFTA00911631':'obey the rules',
'EFTA02449383':'what you are told','EFTA00958296':'need you','EFTA02455750':'deception','EFTA01948560':'falsehoods','EFTA02446419':'deception',
'EFTA01009902':'Cheating','EFTA00741074':'repeated game','EFTA02651127':'cheating','EFTA01780849':'fasb','EFTA00818018':'sanctions','EFTA00881507':'church structure',
'EFTA00857669':'joke','EFTA00857538':'kidding','EFTA01053588':'sense of humor','EFTA01851099':'teasing','EFTA00964951':'shoot myself',
'EFTA01976625':'not fast','EFTA02572371':'virus','EFTA02715921':'function of want','EFTA00854413':'social understanding','EFTA02461213':'promise',
'EFTA00827784':'predator prey','EFTA00896136':'chemistry','EFTA00967134':'reductionism','EFTA01948967':'statistical mechanics','EFTA00874534':'bupkis',
'EFTA01042028':'blah','EFTA01956464':'sophis','EFTA02524708':'scientists','EFTA02436347':'smell','EFTA02564253':'dreams occur','TEXT-001-HOUSE_OVERSIGHT_029085':'deception',
'EFTA01951621':'circadian','EFTA01983296':'displace','EFTA02417871':'google page','EFTA00963770':'reputation changer','EFTA00838586':'coherent presentation',
'EFTA00822873':'get ahead','EFTA00817973':'book signing','EFTA02345513':'documentary','EFTA02632540':'pr person','EFTA02632734':'journal','EFTA02635173':'solitary',
'HOUSE_OVERSIGHT_031676':'ludicrous','EFTA02632614':'chip away','EFTA01029328':'punching bag','EFTA02670320':'solitary','EFTA02590018':'solitary',
'EFTA00997902':'solitary','EFTA00696252':'solitary','EFTA00915181':'billionaire','EFTA00949206':'billion dollars','EFTA01008505':'media lab','EFTA00895144':'500,000',
'EFTA02596421':'3,000,000','EFTA01818555':'backround','EFTA02580943':'press','EFTA00711754':'prize','EFTA02452507':'shame','EFTA02639355':'colorful past',
'EFTA00837628':'true friends','EFTA01739467':'shrink','EFTA00740108':'are you ok','EFTA00903286':'thanks','EFTA00840803':'whining','EFTA00913146':'misdirection',
'EFTA01819482':'Medev','EFTA02525522':'slogans','EFTA01781638':'ugly messages','EFTA01811429':'weight','EFTA02336803':'weight','EFTA00699155':'face outward',
'EFTA01941232':'bitcoin','EFTA01923124':'currency','EFTA01930285':'ukraine','EFTA01977138':'campaign','EFTA01041630':'terrorism','EFTA00819060':'social media',
'EFTA01820466':'face','TEXT-001-HOUSE_OVERSIGHT_029585':'hyatt','EFTA00855008':'politics','EFTA02261563':'board','EFTA00901553':'private person','EFTA02636272':'ego',
'EFTA00694632':'jail','EFTA00490011':'marrakesh','EFTA02282669':'zorro','EFTA00495115':'paris','EFTA02636690':'kuwait','EFTA00471386':'no cell','EFTA02333309':'paris time',
'EFTA02642963':'paris time','EFTA02633272':'vi time','EFTA02333127':'bulletproof','EFTA02532439':'prescription','EFTA00741490':'last memory','EFTA01037762':'lauder',
}
bad=[]
for i,ph in checks.items():
    pat=re.escape(i)+r'\|[^\n]*'
    hit=[l for l in re.findall(pat,txt) if ph.lower() in l.lower()]
    if not hit:
        b=' '.join(allm[allm.doc_id==i].body.fillna('').tolist())
        if ph.lower() not in b.lower(): bad.append((i,ph, len(b)))
print('checked',len(checks),'not confirmed:',bad)
