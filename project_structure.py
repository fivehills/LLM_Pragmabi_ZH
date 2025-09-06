# config.json.template
{
  "openrouter_api_key": "your-openrouter-api-key-here",
  "openai_api_key": "your-openai-api-key-here",
  "anthropic_api_key": "your-anthropic-api-key-here",
  "default_models": ["gpt-4o-mini", "claude-3-haiku"],
  "max_retries": 3,
  "request_timeout": 60,
  "rate_limit_delay": 1.0,
  "evaluation_settings": {
    "default_sample_size": 50,
    "enable_arbitration": true,
    "export_detailed_results": true
  }
}

# requirements.txt
requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# datasets/sarcasm/sarcasm_samples.json
{
  "metadata": {
    "description": "Chinese sarcasm and irony detection dataset",
    "total_samples": 15,
    "sarcastic_samples": 10,
    "non_sarcastic_samples": 5,
    "source": "Curated from social media and literature"
  },
  "samples": [
    {"text": "这是最受气包的一届总统特朗普:我被陷害,我被诬告,我冤枉!特朗普反对者:我们只是在行使民主自由的权利!!", "label": 0, "type": "政治评论"},
    {"text": "那些相信经济全球化会走向命运共同体的人现在天天被打脸。", "label": 0, "type": "经济观点"},
    {"text": "嗯,这很西方,有强盗思维逻辑", "label": 0, "type": "文化评价"},
    {"text": "呵呵呵呵,站着说话不腰疼,棚户区改造你们怎么不说,穷人不是人?", "label": 0, "type": "社会批评"},
    {"text": "波音:世界末日到了空客:武林盟主是我的了:这是百年难遇的机会", "label": 0, "type": "商业调侃"},
    {"text": "我最喜欢阿柳一本正经地胡说八道啦。", "label": 1, "type": "人物调侃"},
    {"text": "特朗普这个宣传员做的不错。打脸美国所谓的自由民主。", "label": 1, "type": "政治讽刺"},
    {"text": "令人震惊的美国特色......", "label": 1, "type": "国际讽刺"},
    {"text": "黄鼠狼给鸡拜年,没安好心。", "label": 1, "type": "成语讽刺"},
    {"text": "哈哈,骚操作啊", "label": 0, "type": "网络用语"},
    {"text": "你真是太聪明了，把简单的问题复杂化", "label": 1, "type": "能力讽刺"},
    {"text": "这种天气真适合出门，狂风暴雨的", "label": 1, "type": "天气讽刺"},
    {"text": "他的演讲水平真高，台下观众都睡着了", "label": 1, "type": "技能讽刺"},
    {"text": "这家餐厅的服务真周到，等了两个小时才上菜", "label": 1, "type": "服务讽刺"},
    {"text": "今天的会议非常高效，讨论了三个小时没有结论", "label": 1, "type": "效率讽刺"}
  ]
}

# datasets/idioms/idiom_translation_pairs.json
{
  "metadata": {
    "description": "Chinese idiom to English translation dataset",
    "total_pairs": 20,
    "categories": ["efficiency", "determination", "limitation", "assistance"],
    "source": "Academic linguistic research and bilingual dictionaries"
  },
  "idioms": [
    {"chinese": "一石二鸟", "english": "kill two birds with one stone", "category": "efficiency", "difficulty": "easy"},
    {"chinese": "破釜沉舟", "english": "burn one's bridges", "category": "determination", "difficulty": "medium"},
    {"chinese": "画蛇添足", "english": "gild the lily", "category": "excess", "difficulty": "medium"},
    {"chinese": "井底之蛙", "english": "a frog in a well", "category": "limitation", "difficulty": "easy"},
    {"chinese": "雪中送炭", "english": "timely help", "category": "assistance", "difficulty": "medium"},
    {"chinese": "亡羊补牢", "english": "better late than never", "category": "correction", "difficulty": "medium"},
    {"chinese": "守株待兔", "english": "wait for windfalls", "category": "passivity", "difficulty": "hard"},
    {"chinese": "掩耳盗铃", "english": "bury one's head in the sand", "category": "self-deception", "difficulty": "medium"},
    {"chinese": "南辕北辙", "english": "work at cross purposes", "category": "contradiction", "difficulty": "hard"},
    {"chinese": "杯弓蛇影", "english": "be frightened by false alarms", "category": "fear", "difficulty": "hard"},
    {"chinese": "班门弄斧", "english": "teach fish to swim", "category": "presumption", "difficulty": "medium"},
    {"chinese": "塞翁失马", "english": "blessing in disguise", "category": "fortune", "difficulty": "medium"},
    {"chinese": "望梅止渴", "english": "feed on illusions", "category": "self-consolation", "difficulty": "hard"},
    {"chinese": "叶公好龙", "english": "profess to like what one fears", "category": "hypocrisy", "difficulty": "hard"},
    {"chinese